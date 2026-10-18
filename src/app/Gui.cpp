// GUI of the application (ImGui + ImPlot): a four-stage workflow (Map, Terrain, Conditions, Simulation) with a
// stage bar at the top, one side panel whose content follows the stage, the view in the remaining space and a
// plots band under it in the Simulation stage. Fixed layout, computed from the viewport every frame.
// The frame loop is in App.cpp.
#include "app/App.h"

#include "util/Log.h"
#include "util/Paths.h"

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_internal.h>
#include <implot.h>

#include <glm/gtc/matrix_inverse.hpp>

#include <algorithm>
#include <cmath>

namespace swe {

namespace {

constexpr float kSidePanelWidth = 390.0f;
constexpr float kStageBarHeight = 46.0f;

constexpr ImGuiWindowFlags kFixedWindow = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                                          ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoSavedSettings |
                                          ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;

struct StageInfo
{
	char const *number;
	char const *label;
	char const *hint;
};
constexpr StageInfo kStages[] = {
    {"1", "Carte", "Choisir une zone sur la carte et importer son relief, ou ouvrir un cas enregistré"},
    {"2", "Terrain", "Le relief en 3D : exagération verticale, orthophoto, retouches du lit"},
    {"3", "Conditions", "Eau initiale, sol et nappe, pluie, zones et conditions aux limites, sur le maillage"},
    {"4", "Simulation", "Lancer, suivre le bilan, colorer, sonder, tracer des courbes"},
};

// probe colours (also used by the plots and the 3D markers)
ImVec4 probeColor(size_t p)
{
	static ImVec4 const colors[MAX_PROBES] = {{0.95f, 0.35f, 0.25f, 1}, {0.25f, 0.65f, 0.95f, 1}, {0.35f, 0.85f, 0.35f, 1},
	                                          {0.95f, 0.75f, 0.20f, 1}, {0.75f, 0.45f, 0.95f, 1}, {0.20f, 0.85f, 0.80f, 1},
	                                          {0.95f, 0.50f, 0.75f, 1}, {0.70f, 0.70f, 0.70f, 1}};
	return colors[p % MAX_PROBES];
}

void tip(char const *text)
{
	if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort | ImGuiHoveredFlags_AllowWhenDisabled))
		ImGui::SetTooltip("%s", text);
}

std::string formatTime(double t)
{
	if (!std::isfinite(t) || t < 0.0 || t > 1e9)
		return "—";
	if (t < 120.0)
		return std::format("{:.2f} s", t);
	int s = static_cast<int>(t);
	if (s < 3600)
		return std::format("{} min {:02d} s", s / 60, s % 60);
	return std::format("{} h {:02d} min {:02d} s", s / 3600, (s / 60) % 60, s % 60);
}

std::string formatArea(double a)
{
	return a >= 1e5 ? std::format("{:.3f} km²", a * 1e-6) : std::format("{:.1f} m²", a);
}

std::string formatVolume(double v)
{
	if (v >= 1e6)
		return std::format("{:.4g} hm³", v * 1e-6);
	return std::format("{:.4g} m³", v);
}

struct ColorModeInfo
{
	ColorMode   mode;
	char const *label;
	char const *unit;
	enum Needs { Always, Aquifer, Chronology } needs = Always;        // ground quantities need an aquifer / a chronology
};
constexpr ColorModeInfo kColorModes[] = {
    {ColorMode::Realistic, "Réaliste", ""},
    {ColorMode::Depth, "Hauteur d'eau h", "m"},
    {ColorMode::Speed, "Vitesse |u|", "m/s"},
    {ColorMode::Froude, "Nombre de Froude", ""},
    {ColorMode::Elevation, "Cote de la surface libre", "m"},
    {ColorMode::GwTableDepth, "Nappe : profondeur sous le sol", "m", ColorModeInfo::Aquifer},
    {ColorMode::GwSaturation, "Nappe : saturation du sol", "", ColorModeInfo::Aquifer},
    {ColorMode::GwExchange, "Nappe : échange (+ infiltration)", "mm/h", ColorModeInfo::Aquifer},
    {ColorMode::GwInfiltrated, "Nappe : lame infiltrée nette", "mm", ColorModeInfo::Aquifer},
    {ColorMode::GwHead, "Nappe : cote piézométrique", "m", ColorModeInfo::Aquifer},
    {ColorMode::Rain, "Pluie cumulée", "mm", ColorModeInfo::Chronology},
};
static_assert(std::size(kColorModes) == size_t(ColorMode::Count));

bool available(ColorModeInfo const &m, CaseData const &c)
{
	return m.needs == ColorModeInfo::Always || (m.needs == ColorModeInfo::Aquifer && c.gw.enabled) ||
	       (m.needs == ColorModeInfo::Chronology && (c.infiltration.active() || c.rain.active()));
}

ColorModeInfo const &colorModeInfo(ColorMode m)
{
	return kColorModes[static_cast<int>(m)];
}

// Prominent full-width action button (the "next step" of each stage)
bool primaryButton(char const *label, float height = 0.0f)
{
	ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.16f, 0.48f, 0.78f, 1.0f));
	ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.22f, 0.58f, 0.90f, 1.0f));
	ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.12f, 0.40f, 0.68f, 1.0f));
	bool pressed = ImGui::Button(label, ImVec2(-1, height));
	ImGui::PopStyleColor(3);
	return pressed;
}

std::filesystem::path capturePath(char const *prefix)
{
	return paths::dataDir() / "captures" /
	       std::format("{}_{:%Y%m%d_%H%M%S}.png", prefix, std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now()));
}

}        // namespace

// ---------------------------------------------------------------------------------------------
// Frame layout
// ---------------------------------------------------------------------------------------------

void App::drawGui()
{
	if (autoRange_)        // the legend stays in presentation mode: keep its range live there too
		autoColorRange();
	auto  *vp = ImGui::GetMainViewport();
	ImVec2 sc = ImGui::GetIO().DisplayFramebufferScale;
	auto   toPixels = [&](ImVec2 pos, ImVec2 size) {
        return glm::vec4((pos.x - vp->Pos.x) * sc.x, (pos.y - vp->Pos.y) * sc.y, size.x * sc.x, size.y * sc.y);
	};
	if (hideGui_)
	{
		// presentation mode: the 3D view fills the window, legend and probe markers stay
		viewRect_ = glm::vec4(0.0f, 0.0f, vp->Size.x * sc.x, vp->Size.y * sc.y);
		if (stage_ != Stage::Map)
		{
			drawLegend();
			drawProbeMarkers();
		}
		return;
	}
	drawMenuBar();
	float  menuH = ImGui::GetFrameHeight();
	ImVec2 origin(vp->Pos.x, vp->Pos.y + menuH);
	ImVec2 work(vp->Size.x, vp->Size.y - menuH);
	drawStageBar(origin.y, kStageBarHeight);

	float  panelW = std::min(kSidePanelWidth, std::round(work.x * 0.42f));
	ImVec2 panelPos(origin.x, origin.y + kStageBarHeight), panelSize(panelW, work.y - kStageBarHeight);
	ImVec2 viewPos(origin.x + panelW, panelPos.y), viewSize(work.x - panelW, panelSize.y);
	bool   plots  = stage_ == Stage::Simulation && showPlots_;
	float  plotsH = plots ? std::round(viewSize.y * 0.30f) : 0.0f;
	viewSize.y -= plotsH;

	drawSidePanel(panelPos, panelSize);
	viewRect_ = toPixels(viewPos, viewSize);
	if (stage_ == Stage::Map || transition_.active)
		drawMapStage(viewPos, viewSize);
	if (stage_ != Stage::Map)
	{
		if (plots)
			drawPlotsBand(ImVec2(viewPos.x, viewPos.y + viewSize.y), ImVec2(viewSize.x, plotsH));
		if (!transition_.active)
		{
			drawLegend();
			drawProbeMarkers();
		}
	}
	if (showPerf_)
		drawPerfWindow();
	if (showHelp_)
		drawHelp();
	if (showDemo_)
		ImGui::ShowDemoWindow(&showDemo_);
}

void App::drawMenuBar()
{
	if (!ImGui::BeginMainMenuBar())
		return;
	if (ImGui::BeginMenu("Fichier"))
	{
		if (ImGui::MenuItem("Capture d'écran", "F12"))
			captureRequest_ = capturePath("swe");
		if (ImGui::MenuItem(recording_ ? "Arrêter l'enregistrement" : "Enregistrer une séquence d'images"))
		{
			recording_ = !recording_;
			if (recording_)
			{
				recordDir_   = paths::dataDir() / "captures" /
				             std::format("rec_{:%Y%m%d_%H%M%S}", std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now()));
				recordIndex_ = 0;
				std::filesystem::create_directories(recordDir_);
				log::info("recording to {}", std::filesystem::absolute(recordDir_).string());
			}
		}
		ImGui::SetNextItemWidth(120);
		ImGui::SliderInt("1 image sur", &recordEvery_, 1, 10);
		tip("Séquence PNG dans captures/rec_*/ (assembler avec : ffmpeg -framerate 30 -i frame_%05d.png out.mp4)");
		ImGui::Separator();
		if (caseGeo_ && ImGui::MenuItem("Enregistrer le cas"))
			saveCurrentCase();
		ImGui::Separator();
		if (ImGui::MenuItem("Quitter", "Alt+F4"))
			glfwSetWindowShouldClose(window_, GLFW_TRUE);
		ImGui::EndMenu();
	}
	if (ImGui::BeginMenu("Affichage"))
	{
		ImGui::MenuItem("Courbes (étape Simulation)", nullptr, &showPlots_);
		ImGui::MenuItem("Performances", nullptr, &showPerf_);
		if (ImGui::MenuItem("Interface masquée", "F11", hideGui_))
			hideGui_ = !hideGui_;
		ImGui::Separator();
		if (ImGui::MenuItem("Recadrer la caméra", "Origine"))
			animateCamera(camera_.home());
		ImGui::MenuItem("Démo ImGui", nullptr, &showDemo_);
		ImGui::EndMenu();
	}
	if (ImGui::BeginMenu("Aide"))
	{
		ImGui::MenuItem("Raccourcis et souris", "F1", &showHelp_);
		ImGui::EndMenu();
	}
	ImGui::EndMainMenuBar();
}

// Stage bar: the four steps of the workflow, the current one highlighted; status on the right
void App::drawStageBar(float y, float height)
{
	auto *vp = ImGui::GetMainViewport();
	ImGui::SetNextWindowPos(ImVec2(vp->Pos.x, y));
	ImGui::SetNextWindowSize(ImVec2(vp->Size.x, height));
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10.0f, 7.0f));
	ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
	ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.09f, 0.10f, 0.12f, 1.0f));
	if (ImGui::Begin("##stagebar", nullptr, kFixedWindow | ImGuiWindowFlags_NoScrollbar))
	{
		bool loaded = session_->loaded();
		for (int k = 0; k < 4; ++k)
		{
			auto  s      = static_cast<Stage>(k);
			bool  active = s == stage_;
			bool  enabled = k == 0 || loaded;
			auto  label  = std::format("{}  {}", kStages[k].number, kStages[k].label);
			ImGui::PushID(k);
			if (active)
			{
				ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.16f, 0.48f, 0.78f, 1.0f));
				ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.16f, 0.48f, 0.78f, 1.0f));
				ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.16f, 0.48f, 0.78f, 1.0f));
			}
			else
			{
				ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.16f, 0.18f, 0.21f, 1.0f));
				ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.24f, 0.27f, 0.31f, 1.0f));
				ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.20f, 0.23f, 0.27f, 1.0f));
			}
			ImGui::BeginDisabled(!enabled);
			if (ImGui::Button(label.c_str(), ImVec2(150.0f, height - 14.0f)) && !active)
				setStage(s);
			ImGui::EndDisabled();
			ImGui::PopStyleColor(3);
			tip(kStages[k].hint);
			ImGui::PopID();
			if (k < 3)
			{
				ImGui::SameLine(0.0f, 6.0f);
				ImGui::TextDisabled(">");
				ImGui::SameLine(0.0f, 6.0f);
			}
		}

		// status: import in progress, else case / clock / pause
		std::string status;
		if (map_->importing())
			status = std::format("Import... {}  {:.0f} %", map_->progressStage(), 100.0f * map_->progressFraction());
		else
		{
			auto const &c = session_->caseData();
			status        = std::format("{}   t = {}{}{}", c.label, formatTime(scalars_.t),
			                            stage_ == Stage::Simulation ? (paused_ ? "   EN PAUSE" : "") : "   (t = 0, en pause)",
			                            recording_ ? "   ● REC" : "");
		}
		float w = ImGui::CalcTextSize(status.c_str()).x;
		ImGui::SameLine(std::max(ImGui::GetCursorPosX() + 20.0f, ImGui::GetWindowWidth() - w - 16.0f));
		ImGui::AlignTextToFramePadding();
		ImGui::TextUnformatted(status.c_str());
	}
	ImGui::End();
	ImGui::PopStyleColor();
	ImGui::PopStyleVar(2);
}

// Side panel: the stage's controls
void App::drawSidePanel(ImVec2 pos, ImVec2 size)
{
	ImGui::SetNextWindowPos(pos);
	ImGui::SetNextWindowSize(size);
	ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12.0f, 10.0f));
	if (ImGui::Begin("##side", nullptr, kFixedWindow))
	{
		switch (stage_)
		{
		case Stage::Map:
		{
			ImGui::SeparatorText("Cas enregistrés");
			auto const &current = session_->caseData().name;
			float       rows    = std::clamp(float(caseList_.size()), 2.0f, 7.0f);
			if (ImGui::BeginListBox("##cases", ImVec2(-1, rows * ImGui::GetTextLineHeightWithSpacing() + 6.0f)))
			{
				for (auto const &info : caseList_)
				{
					bool selected = info.name == current;
					if (ImGui::Selectable(info.label.c_str(), selected, ImGuiSelectableFlags_AllowDoubleClick) &&
					    ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))
						openCase(info.name);
					if (ImGui::IsItemHovered())
						ImGui::SetTooltip("%s  (double-clic : ouvrir)", info.name.c_str());
					if (ImGui::IsItemClicked() && !selected)
					{
						loadCase(info.name);        // shows its footprint on the map
						setStage(Stage::Map);
					}
				}
				ImGui::EndListBox();
			}
			if (primaryButton(std::format("Ouvrir « {} »  >", session_->caseData().label).c_str()))
				openCase(current);
			tip("Charge le cas et montre son terrain en 3D");
			ImGui::Spacing();
			map_->drawImportSidebar();
			break;
		}
		case Stage::Terrain: drawTerrainPanel(); break;
		case Stage::Conditions: drawConditionsPanel(); break;
		case Stage::Simulation: drawSimulationPanel(); break;
		}
	}
	ImGui::End();
	ImGui::PopStyleVar(2);
}

// Map stage: the 2D map fills the view (also drawn fading out during the transition to the 3D view)
void App::drawMapStage(ImVec2 pos, ImVec2 size)
{
	ImGui::SetNextWindowPos(pos);
	ImGui::SetNextWindowSize(size);
	ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8.0f, 6.0f));
	ImGuiWindowFlags flags = kFixedWindow | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse;
	if (transition_.active)
	{
		ImGui::SetNextWindowBgAlpha(0.0f);
		flags |= ImGuiWindowFlags_NoInputs;
	}
	if (ImGui::Begin("##map", nullptr, flags))
		map_->drawFull(frameCount_);
	ImGui::End();
	ImGui::PopStyleVar(2);
}

// ---------------------------------------------------------------------------------------------
// Stage 2: Terrain
// ---------------------------------------------------------------------------------------------

void App::drawCameraButtons()
{
	float bw = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) * 0.5f;
	if (ImGui::Button("Vue du dessus", ImVec2(bw, 0)))
		animateCamera(topPose());
	tip("Caméra à la verticale, nord en haut (molette : zoom, clic milieu : déplacer)");
	ImGui::SameLine();
	if (ImGui::Button("Vue 3D", ImVec2(bw, 0)))
		animateCamera(camera_.home());
	tip("Origine");
}

void App::drawTerrainPanel()
{
	auto const &c = session_->caseData();
	ImGui::SeparatorText("Terrain");
	ImGui::TextWrapped("%s", c.label.c_str());
	ImGui::TextDisabled("%u x %u mailles de %.3g m  (%.2f M)", c.nx, c.ny, c.dx(), c.cells() * 1e-6);
	ImGui::TextDisabled("%.0f x %.0f m, z_ref = %.0f m", c.Lx, c.Ly, c.z_ref);
	auto [zmin, zmax] = session_->bedRange();
	ImGui::TextDisabled("relief : %.1f à %.1f m DNG", zmin + c.z_ref, zmax + c.z_ref);
	if (transition_.active)
		ImGui::TextColored(ImVec4(0.6f, 0.8f, 1.0f, 1.0f), "Extrusion du relief... (clic pour passer)");

	ImGui::SeparatorText("Vue");
	drawCameraButtons();
	ImGui::SetNextItemWidth(-110);
	ImGui::SliderFloat("exagération", &render_.z_scale, 1.0f, 2.5f * zScale0_, "x %.2f");
	tip("Exagération verticale du relief (les ombres suivent)");
	if (renderer_->hasOrthophoto())
	{
		ImGui::Checkbox("Orthophoto", &render_.showOrtho);
		if (render_.showOrtho)
		{
			ImGui::SameLine();
			ImGui::SetNextItemWidth(-110);
			ImGui::SliderFloat("mélange", &render_.orthoMix, 0.0f, 1.0f, "%.2f");
		}
	}
	ImGui::Checkbox("Maillage", &meshPreference_);
	render_.showMesh = meshPreference_;

	ImGui::SeparatorText("Relief");
	if (caseGeo_)
	{
		ImGui::TextWrapped("Le LiDAR mesure la surface de l'eau : là où aucune bathymétrie n'existe, le lit de la rivière se "
		                   "creuse sur la carte (axe + largeur + profondeur).");
		if (ImGui::Button("Modifier le lit sur la carte", ImVec2(-1, 0)))
			setStage(Stage::Map);
		tip("Retour à la carte : section « Avancé » > « Lit de rivière »");
	}
	else
		ImGui::TextDisabled("Cas analytique : relief fixé par le cas.");

	// footer
	float footer = ImGui::GetFrameHeightWithSpacing() * 1.6f;
	ImGui::SetCursorPosY(std::max(ImGui::GetCursorPosY(), ImGui::GetWindowHeight() - footer));
	if (primaryButton("Suivant : Conditions  >", ImGui::GetFrameHeight() * 1.3f))
		setStage(Stage::Conditions);
}

// ---------------------------------------------------------------------------------------------
// Stage 3: Conditions
// ---------------------------------------------------------------------------------------------

void App::drawSelectionHeader()
{
	auto const &c   = session_->caseData();
	auto        sel = render_.selection;
	bool hasSel     = sel.z >= sel.x && sel.w >= sel.y;
	ImGui::SeparatorText("Sélection");
	if (hasSel)
	{
		uint32_t ni = sel.z - sel.x + 1, nj = sel.w - sel.y + 1;
		ImGui::Text("mailles (%u..%u, %u..%u) : %u x %u = %u", sel.x, sel.z, sel.y, sel.w, ni, nj, ni * nj);
		float zmin = 1e30f, zmax = -1e30f;
		for (uint32_t i = sel.x; i <= sel.z; ++i)
			for (uint32_t j = sel.y; j <= sel.w; ++j)
			{
				float z = c.zb[size_t(i) * c.ny + j];
				zmin    = std::min(zmin, z);
				zmax    = std::max(zmax, z);
			}
		ImGui::TextDisabled("%.0f x %.0f m, fond %.2f à %.2f m DNG", ni * c.dx(), nj * c.dy(), zmin + c.z_ref, zmax + c.z_ref);
	}
	else
		ImGui::TextWrapped("Glisser dans la vue : rectangle de mailles (Échap : annuler).");
	if (ImGui::SmallButton("Tout le domaine"))
	{
		selAnchor_        = Probe{0, 0};
		selCurrent_       = Probe{c.nx - 1, c.ny - 1};
		render_.selection = {0, 0, c.nx - 1, c.ny - 1};
	}
	ImGui::SameLine();
	ImGui::BeginDisabled(!hasSel);
	if (ImGui::SmallButton("Effacer"))
	{
		selAnchor_.reset();
		selCurrent_.reset();
		render_.selection = {1, 0, 0, 0};
	}
	ImGui::EndDisabled();
	drawCameraButtons();
}

void App::drawWaterTab()
{
	auto const &c      = session_->caseData();
	auto        sel    = render_.selection;
	bool        hasSel = sel.z >= sel.x && sel.w >= sel.y;
	ImGui::SeparatorText("Bloc d'eau sur la sélection");
	ImGui::SetNextItemWidth(-110);
	ImGui::Combo("mode", &waterMode_, "hauteur d'eau (m)\0cote de surface (m DNG)\0ajouter (m)\0");
	ImGui::SetNextItemWidth(-110);
	ImGui::InputFloat("valeur", &waterValue_, 0.1f, 1.0f, "%.2f");
	ImGui::BeginDisabled(!hasSel);
	if (ImGui::Button("Poser l'eau", ImVec2(-80, 0)))
	{
		WaterEdit mode  = waterMode_ == 0 ? WaterEdit::Depth : waterMode_ == 1 ? WaterEdit::Level : WaterEdit::Add;
		float     value = waterMode_ == 1 ? waterValue_ - static_cast<float>(c.z_ref) : waterValue_;
		session_->editWater(selection(), mode, value, true);
		effects_->reset();
	}
	ImGui::SameLine();
	if (ImGui::Button("Vider"))
	{
		session_->editWater(selection(), WaterEdit::Clear, 0.0f, true);
		effects_->reset();
	}
	ImGui::EndDisabled();
	tip("Eau au repos dans la sélection ; fait partie de l'état initial (Recommencer la restaure)");

	ImGui::SeparatorText("Niveau initial sur tout le domaine");
	if (levelValue_ == 0.0)
		levelValue_ = c.eta_0 + c.z_ref;
	ImGui::SetNextItemWidth(-110);
	ImGui::InputDouble("cote (m DNG)", &levelValue_, 0.5, 5.0, "%.2f");
	if (ImGui::Button("Remplir jusqu'à cette cote", ImVec2(-1, 0)))
	{
		session_->editWater({0, 0, c.nx - 1, c.ny - 1}, WaterEdit::Level, static_cast<float>(levelValue_ - c.z_ref), true);
		effects_->reset();
	}
	tip("Lac au repos : h = max(cote - fond, 0) partout");
	ImGui::Checkbox("Eau mince visible", &render_.thinWater);
	tip("Donne une opacité minimale aux lames d'eau de quelques cm, sinon invisibles");
}

void App::drawSoilTab()
{
	auto const &c  = session_->caseData();
	bool        on = c.gw.enabled;
	ImGui::SeparatorText("Nappe souterraine");
	if (ImGui::Checkbox("Aquifère sous le terrain", &on))
	{
		session_->setGroundwater(on ? std::optional(soil_) : std::nullopt);
		onCaseLoaded();
		updateProbeList();
		profileCells_.clear();
	}
	tip("Échange vertical avec l'eau de surface (infiltration / résurgence) et écoulement latéral de Boussinesq,\n"
	    "comme la couche souterraine de wolfgpu. Le cas est rechargé à t = 0.");
	if (c.gw.wolfgpu)
		ImGui::TextDisabled("Convention wolfgpu (Sy = 1, aquifère posé sur le lit) : définie par le fichier du cas.");

	ImGui::SeparatorText(on ? "Paramètres du sol" : "Paramètres du sol (à l'activation)");
	auto param = [&](char const *label, double &v, double step, double big, char const *fmt, char const *help) {
		ImGui::SetNextItemWidth(-130);
		bool changed = ImGui::InputDouble(label, &v, step, big, fmt);
		tip(help);
		return changed;
	};
	param("kv (m/s)", soil_.kv, 1e-6, 1e-5, "%.2e", "Conductivité verticale : vitesse d'infiltration de référence");
	param("kh (m/s)", soil_.kh, 1e-5, 1e-4, "%.2e", "Conductivité horizontale : écoulement latéral de la nappe (0 = aucun)");
	param("Sy", soil_.sy, 0.05, 0.1, "%.2f", "Porosité de drainage (0,01 argile ... 0,3 sable)");
	param("épaisseur (m)", soil_.depth, 0.5, 2.0, "%.2f", "Épaisseur du sol au-dessus du substratum imperméable");
	param("nappe à (m)", soil_.tableDepth, 0.25, 1.0, "%.2f", "Profondeur initiale de la nappe sous le sol (0 = saturé)");
	if (on)
	{
		auto sel    = render_.selection;
		bool hasSel = sel.z >= sel.x && sel.w >= sel.y;
		if (ImGui::Button("Appliquer partout", ImVec2(-150, 0)))
		{
			session_->setSoil({0, 0, c.nx - 1, c.ny - 1}, soil_);
			onCaseLoaded();
		}
		ImGui::SameLine();
		ImGui::BeginDisabled(!hasSel);
		if (ImGui::Button("À la sélection", ImVec2(-1, 0)))
		{
			session_->setSoil(selection(), soil_);
			onCaseLoaded();
		}
		ImGui::EndDisabled();
		tip("Sol hétérogène : des valeurs différentes sur un rectangle de mailles");
	}

	ImGui::SeparatorText("Frottement");
	float manning = static_cast<float>(c.manning_n);
	ImGui::SetNextItemWidth(-130);
	if (ImGui::SliderFloat("Manning n", &manning, 0.0f, 0.1f, "%.3f"))
		session_->setManning(manning);
	tip("Frottement de fond (s/m^1/3) : 0.03 lit naturel, 0.04 plaine, 0 sans frottement");
}

void App::drawZonesTab()
{
	auto const &c      = session_->caseData();
	auto        sel    = render_.selection;
	bool        hasSel = sel.z >= sel.x && sel.w >= sel.y;
	ImGui::SeparatorText("Nouvelle zone sur la sélection");
	ImGui::TextDisabled("Appliquée à la fin de chaque pas de temps.");
	ImGui::SetNextItemWidth(-110);
	ImGui::Combo("type", &zoneKind_, "source : débit (m3/s)\0puits (exutoire)\0niveau fixé (m DNG)\0");
	if (zoneKind_ != 1)
	{
		ImGui::SetNextItemWidth(-110);
		ImGui::InputFloat("##zval", &zoneValue_, 1.0f, 10.0f, zoneKind_ == 0 ? "%.2f m3/s" : "%.2f m DNG");
	}
	ImGui::SetNextItemWidth(-110);
	ImGui::InputTextWithHint("nom", "optionnel", zoneLabel_, sizeof(zoneLabel_));
	ImGui::BeginDisabled(!hasSel || c.zones.size() >= MAX_ZONES);
	if (ImGui::Button("Créer la zone", ImVec2(-1, 0)))
	{
		ZoneKind kind  = zoneKind_ == 0 ? ZoneKind::Source : zoneKind_ == 1 ? ZoneKind::Sink : ZoneKind::Level;
		double   value = zoneKind_ == 2 ? zoneValue_ - c.z_ref : zoneValue_;
		std::string label = zoneLabel_[0] ? zoneLabel_ : (zoneKind_ == 0 ? "source" : zoneKind_ == 1 ? "puits" : "niveau");
		session_->addZone(selection(), kind, value, label);
		zoneLabel_[0] = 0;
	}
	ImGui::EndDisabled();
	if (!c.zones.empty())
		ImGui::SeparatorText("Zones du cas");
	if (!c.zones.empty() && ImGui::BeginTable("zones", 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp))
	{
		ImGui::TableSetupColumn("zone");
		ImGui::TableSetupColumn("type", ImGuiTableColumnFlags_WidthFixed, 50);
		ImGui::TableSetupColumn("valeur", ImGuiTableColumnFlags_WidthFixed, 110);
		ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 22);
		ImGui::TableHeadersRow();
		int remove = -1;
		for (size_t z = 0; z < c.zones.size(); ++z)
		{
			auto const &zd = c.zones[z];
			ImGui::PushID(int(z));
			ImGui::TableNextRow();
			ImGui::TableNextColumn();
			ImVec4 col = zd.kind == ZoneKind::Source ? ImVec4(0.35f, 0.6f, 1, 1) : zd.kind == ZoneKind::Sink ? ImVec4(1, 0.4f, 0.3f, 1) : ImVec4(0.3f, 0.95f, 0.9f, 1);
			ImGui::TextColored(col, "%s", zd.label.c_str());
			if (ImGui::IsItemHovered())
				ImGui::SetTooltip("mailles (%u..%u, %u..%u)", zd.i0, zd.i1, zd.j0, zd.j1);
			ImGui::TableNextColumn();
			ImGui::TextUnformatted(zd.kind == ZoneKind::Source ? "source" : zd.kind == ZoneKind::Sink ? "puits" : "niveau");
			ImGui::TableNextColumn();
			if (zd.kind != ZoneKind::Sink)
			{
				double v = zd.kind == ZoneKind::Level ? zd.value + c.z_ref : zd.value;
				ImGui::SetNextItemWidth(-1);
				if (ImGui::InputDouble("##v", &v, 0.0, 0.0, zd.kind == ZoneKind::Level ? "%.2f m" : "%.1f m3/s", ImGuiInputTextFlags_EnterReturnsTrue))
					session_->setZoneValue(z, zd.kind == ZoneKind::Level ? v - c.z_ref : v);
			}
			ImGui::TableNextColumn();
			if (ImGui::SmallButton("x"))
				remove = int(z);
			ImGui::PopID();
		}
		ImGui::EndTable();
		if (remove >= 0)
			session_->removeZone(size_t(remove));
	}
	ImGui::Spacing();
	ImGui::Checkbox("Afficher les zones dans la vue", &render_.showZones);
	tip("Teinte des zones source / puits / niveau ; hachures des zones de pluie (bleu) et d'infiltration (brun)");
}

void App::drawBoundariesTab()
{
	auto const &c = session_->caseData();
	ImGui::SeparatorText("Bords du domaine");
	static char const *sideNames[4] = {"Ouest", "Est", "Sud", "Nord"};
	static char const *kindNames[3] = {"mur", "débit entrant", "niveau imposé"};
	for (int s = 0; s < 4; ++s)
	{
		ImGui::PushID(s);
		auto bc   = c.bc[s];
		int  kind = static_cast<int>(bc.kind);
		ImGui::SetNextItemWidth(130);
		bool changed = ImGui::Combo(sideNames[s], &kind, kindNames, 3);
		if (changed && kind != static_cast<int>(bc.kind))
		{
			bc.kind  = static_cast<BcKind>(kind);
			bc.value = bc.kind == BcKind::Level ? c.eta_0 : bc.kind == BcKind::Inflow ? 10.0 : 0.0;
		}
		if (bc.kind != BcKind::Wall)
		{
			ImGui::SetNextItemWidth(-1);
			double v = bc.kind == BcKind::Level ? bc.value + c.z_ref : bc.value;
			if (ImGui::InputDouble("##value", &v, 0.0, 0.0, bc.kind == BcKind::Inflow ? "%.1f m3/s" : "%.2f m DNG",
			                       ImGuiInputTextFlags_EnterReturnsTrue))
			{
				bc.value = bc.kind == BcKind::Level ? v - c.z_ref : v;
				changed  = true;
			}
			tip(bc.kind == BcKind::Inflow ? "Débit total réparti sur le bord au prorata de h^(5/3) (Entrée pour valider)"
			                              : "Cote de surface libre imposée (Entrée pour valider)");
		}
		if (changed)
			session_->setBoundary(s, bc);
		ImGui::PopID();
	}
	ImGui::TextDisabled("Un bord fermé est un mur. Les débits entrent par les mailles les plus basses du bord.");
}

void App::drawConditionsPanel()
{
	drawSelectionHeader();
	ImGui::Spacing();
	// tabs fill the panel down to the footer
	float footer = ImGui::GetFrameHeight() * 1.3f + ImGui::GetFrameHeightWithSpacing() + ImGui::GetStyle().ItemSpacing.y * 2;
	if (ImGui::BeginChild("tabs", ImVec2(0, -footer), ImGuiChildFlags_None))
	{
		if (ImGui::BeginTabBar("conditions", ImGuiTabBarFlags_FittingPolicyScroll))
		{
			if (ImGui::BeginTabItem("Eau"))
			{
				drawWaterTab();
				ImGui::EndTabItem();
			}
			if (ImGui::BeginTabItem("Sol"))
			{
				drawSoilTab();
				ImGui::EndTabItem();
			}
			if (ImGui::BeginTabItem("Pluie"))
			{
				drawChronologyEditor();
				ImGui::EndTabItem();
			}
			if (ImGui::BeginTabItem("Zones"))
			{
				drawZonesTab();
				ImGui::EndTabItem();
			}
			if (ImGui::BeginTabItem("Bords"))
			{
				drawBoundariesTab();
				ImGui::EndTabItem();
			}
			ImGui::EndTabBar();
		}
	}
	ImGui::EndChild();
	// footer
	if (caseGeo_)
	{
		if (ImGui::Button("Enregistrer le cas", ImVec2(-1, 0)))
			saveCurrentCase();
		tip("Écrit le lit, l'eau initiale, le sol, les zones et les réglages dans assets/");
	}
	else
		ImGui::TextDisabled("Cas analytique : les conditions ne sont pas enregistrées.");
	if (primaryButton("Lancer la simulation  >>", ImGui::GetFrameHeight() * 1.3f))
		setStage(Stage::Simulation);
}

// Rain / infiltration zones and their table of discharges over time (wolfgpu's add_rain / add_infiltration)
void App::drawChronologyEditor()
{
	using Kind = Session::ChronologyKind;
	auto const &c = session_->caseData();
	ImGui::SeparatorText("Chronologies");
	ImGui::RadioButton("pluie", &chronoKind_, 1);
	ImGui::SameLine();
	ImGui::RadioButton("infiltration", &chronoKind_, 0);
	tip("Débits par zone au cours du temps, comme add_rain / add_infiltration de wolfgpu.\n"
	    "Positif : apport d'eau en surface ; une infiltration négative retire au plus l'eau présente.");
	ImGui::SameLine();
	ImGui::Checkbox("en mm/h", &chronoMmh_);
	tip("Coché : lame d'eau par heure ; décoché : m3/s pour toute la zone (convention wolfgpu)");
	Kind        kind  = chronoKind_ == 1 ? Kind::Rain : Kind::Infiltration;
	auto const &ch    = kind == Kind::Rain ? c.rain : c.infiltration;
	auto        areas = session_->chronologyZoneAreas(kind);
	char const *unit  = chronoMmh_ ? "%.4g mm/h" : "%.4g m3/s";
	auto        toUser   = [&](double q, double area) { return chronoMmh_ ? (area > 0.0 ? q / area * 3.6e6 : 0.0) : q; };
	auto        fromUser = [&](double v, double area) { return chronoMmh_ ? v * area / 3.6e6 : v; };

	// new zone
	auto sel    = render_.selection;
	bool hasSel = sel.z >= sel.x && sel.w >= sel.y;
	ImGui::SetNextItemWidth(-110);
	ImGui::InputFloat("##cval", &chronoValue_, 1.0f, 10.0f, unit);
	CellRect all{0, 0, c.nx - 1, c.ny - 1};
	bool     added = false;
	auto     add = [&](CellRect const &r) {
        double area = double(r.i1 - r.i0 + 1) * (r.j1 - r.j0 + 1) * c.dx() * c.dy();
        session_->addChronologyZone(kind, r, fromUser(chronoValue_, area));
        added = true;
	};
	ImGui::BeginDisabled(!hasSel);
	if (ImGui::Button("Zone = sélection"))
		add(selection());
	ImGui::EndDisabled();
	ImGui::SameLine();
	if (ImGui::Button("Zone = domaine"))
		add(all);
	tip("La valeur s'applique à toutes les lignes de la table ; une maille appartient à une seule zone de chaque type");
	if (added)
		return;        // the zone count and areas computed above are stale: the table is drawn next frame

	if (!ch.active())
	{
		ImGui::TextDisabled("aucune zone de %s", kind == Kind::Rain ? "pluie" : "infiltration");
		return;
	}
	// table: one row per line (start time), one column per zone
	size_t const nz = ch.zoneCount();
	auto         t = ch.t;
	auto         q = ch.q;
	bool         lerp = ch.lerp, changed = false;
	int          removeZone = -1, removeLine = -1;
	int          cols = int(std::min<size_t>(nz, 62)) + 2;
	if (ImGui::BeginTable("chrono", cols, ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollX | ImGuiTableFlags_SizingFixedFit,
	                      ImVec2(0, std::min(30.0f + 24.0f * float(t.size() + 1), 220.0f))))
	{
		ImGui::TableSetupScrollFreeze(1, 1);
		ImGui::TableSetupColumn("t (s)", 0, 70);
		for (int z = 0; z + 2 < cols; ++z)
			ImGui::TableSetupColumn(std::format("zone {}", z + 1).c_str(), 0, 80);
		ImGui::TableSetupColumn("", 0, 22);
		ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
		ImGui::TableNextColumn();
		ImGui::TextUnformatted("t (s)");
		for (int z = 0; z + 2 < cols; ++z)
		{
			ImGui::TableNextColumn();
			ImGui::PushID(z);
			ImGui::Text("zone %d", z + 1);
			if (ImGui::IsItemHovered())
				ImGui::SetTooltip("%.0f m2", areas[z]);
			ImGui::SameLine();
			if (ImGui::SmallButton("x"))
				removeZone = z;
			ImGui::PopID();
		}
		ImGui::TableNextColumn();
		for (size_t l = 0; l < t.size(); ++l)
		{
			ImGui::PushID(int(l));
			ImGui::TableNextRow();
			ImGui::TableNextColumn();
			ImGui::SetNextItemWidth(-1);
			double tl = t[l];
			if (ImGui::InputDouble("##t", &tl, 0.0, 0.0, "%.1f", ImGuiInputTextFlags_EnterReturnsTrue))
			{
				t[l]    = tl;
				changed = true;
			}
			for (int z = 0; z + 2 < cols; ++z)
			{
				ImGui::TableNextColumn();
				ImGui::PushID(z);
				ImGui::SetNextItemWidth(-1);
				double v = toUser(q[l][z], areas[z]);
				if (ImGui::InputDouble("##q", &v, 0.0, 0.0, "%.4g", ImGuiInputTextFlags_EnterReturnsTrue))
				{
					q[l][z] = fromUser(v, areas[z]);
					changed = true;
				}
				ImGui::PopID();
			}
			ImGui::TableNextColumn();
			if (t.size() > 1 && ImGui::SmallButton("x"))
				removeLine = int(l);
			ImGui::PopID();
		}
		ImGui::EndTable();
	}
	if (ImGui::Button("+ ligne"))
	{
		t.push_back(t.back() + 60.0);
		q.push_back(q.back());
		changed = true;
	}
	tip("Nouvelle ligne 60 s après la dernière, mêmes débits (à éditer)");
	ImGui::SameLine();
	if (ImGui::Checkbox("interpolation linéaire", &lerp))
		changed = true;
	tip("Sinon les débits sont constants par paliers ; avant la première ligne : rien, après la dernière : constants");
	ImGui::TextDisabled("valeurs en %s, Entrée pour valider", chronoMmh_ ? "mm/h" : "m3/s par zone");
	if (removeLine >= 0)
	{
		t.erase(t.begin() + removeLine);
		q.erase(q.begin() + removeLine);
		changed = true;
	}
	if (removeZone >= 0)
		session_->removeChronologyZone(kind, size_t(removeZone));
	else if (changed)
	{
		// keep the times increasing: sort the lines by time, drop duplicates
		std::vector<size_t> order(t.size());
		for (size_t k = 0; k < order.size(); ++k)
			order[k] = k;
		std::ranges::stable_sort(order, {}, [&](size_t k) { return t[k]; });
		std::vector<double>              ts;
		std::vector<std::vector<double>> qs;
		for (size_t k : order)
			if (ts.empty() || t[k] > ts.back())
			{
				ts.push_back(t[k]);
				qs.push_back(q[k]);
			}
		session_->setChronologyTable(kind, std::move(ts), std::move(qs), lerp);
	}
}

// ---------------------------------------------------------------------------------------------
// Stage 4: Simulation
// ---------------------------------------------------------------------------------------------

void App::drawTransportSection()
{
	float bw = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) * 0.5f;
	if (paused_)
	{
		if (primaryButton("Lancer  >>", ImGui::GetFrameHeight() * 1.2f))
			paused_ = false;
	}
	else if (ImGui::Button("Pause  ||", ImVec2(-1, ImGui::GetFrameHeight() * 1.2f)))
		paused_ = true;
	tip("Espace");
	if (ImGui::Button("Recommencer", ImVec2(bw, 0)))
		resetSimulation();
	tip("R : remet l'état initial (eau, nappe, zones conservés)");
	ImGui::SameLine();
	if (ImGui::Button("< Conditions", ImVec2(bw, 0)))
		setStage(Stage::Conditions);
	tip("Retour à l'étape Conditions (la simulation repart de t = 0)");
	ImGui::SetNextItemWidth(-90);
	ImGui::SliderInt("pas / image", &stepsPerFrame_, 1, 500, "%d", ImGuiSliderFlags_AlwaysClamp | ImGuiSliderFlags_Logarithmic);
	tip("Nombre de pas de temps calculés entre deux images : plus = simulation plus rapide, image moins fluide");
}

void App::drawBalanceSection()
{
	auto const &c = session_->caseData();
	ImGui::Text("t  = %s   (%u pas)", formatTime(scalars_.t).c_str(), scalars_.n_steps);
	ImGui::Text("dt = %.3e s   s_max = %.3f m/s   %.0f pas/s", scalars_.dt, scalars_.s_max, stepsPerSec_);
	if (!std::isfinite(scalars_.t) || !std::isfinite(scalars_.dt) || !std::isfinite(scalars_.volume))
		ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.3f, 1.0f), "La simulation a divergé : Recommencer");

	ImGui::Spacing();
	double dV = volume0_ > 0.0 ? (scalars_.volume - volume0_) / volume0_ * 100.0 : 0.0;
	ImGui::Text("Volume     %s   (%+.4f %%)", formatVolume(scalars_.volume).c_str(), dV);
	tip("Variation de volume depuis t = 0 : 0 avec des murs partout (conservation), sinon débits entrants / sortants");
	ImGui::Text("Emprise    %s", formatArea(scalars_.wetCells * c.dx() * c.dy()).c_str());
	ImGui::Text("h max      %.3f m     |u| max  %.3f m/s", scalars_.hMax, scalars_.uMax);
	if (c.drain_passes > 0)
		ImGui::Text("Limiteur   %u maille(s) au dernier pas", scalars_.limitedCells);
	else
		ImGui::TextDisabled("Limiteur   désactivé (h < 0 tronqué)");
	tip("Mailles dont les flux sortants ont été réduits pour garder h >= 0 (dernier étage du dernier pas)");
	if (scalars_.clampCells > 0)
		ImGui::Text("h < 0      %u troncature(s), max %.2e m", scalars_.clampCells, scalars_.clampMax);
	tip("Hauteurs négatives ramenées à 0 depuis t = 0 (masse créée). Limiteur actif : seulement des résidus d'arrondi");

	if (c.gw.enabled)
	{
		ImGui::SeparatorText("Nappe");
		ImGui::Text("Volume     %s", formatVolume(scalars_.gwVolume).c_str());
		tip("Eau stockée dans l'aquifère : somme de Sy h_gw dx dy");
		ImGui::Text("Infiltré   %s    exfiltré %s", formatVolume(scalars_.gwInfil).c_str(), formatVolume(scalars_.gwExfil).c_str());
		ImGui::Text("Échange    %+.4g m³/s", scalars_.gwRate);
		tip("Débit net surface -> nappe au dernier pas (> 0 : infiltration)");
	}
	bool chronologies = c.infiltration.active() || c.rain.active();
	if (chronologies)
	{
		ImGui::SeparatorText("Pluie / infiltration");
		ImGui::Text("Pluie      %s    apport net %s", formatVolume(scalars_.srcRain).c_str(), formatVolume(scalars_.srcNet).c_str());
		tip("Volume ajouté à la surface par les chronologies (pluie + infiltration signée), depuis t = 0.\n"
		    "Une infiltration négative ne retire au plus que l'eau présente.");
	}
	if (c.gw.enabled || chronologies)
	{
		double total0 = balance0_;
		double total  = double(scalars_.volume) + scalars_.gwVolume - scalars_.srcNet;
		ImGui::Text("Bilan      %+.4f %% depuis t = 0", total0 > 0.0 ? (total - total0) / total0 * 100.0 : 0.0);
		tip("Surface + nappe - apports des chronologies : 0 avec des murs partout");
	}
}

void App::drawProbesSection()
{
	auto const &c = session_->caseData();
	if (probes_.empty())
		ImGui::TextWrapped("Ctrl + clic sur le terrain pour poser une sonde (8 au plus) : valeurs ici, séries dans les courbes.");
	int             remove     = -1;
	ImGuiTableFlags tableFlags = ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_SizingStretchProp;
	int             cols       = c.gw.enabled ? 6 : 5;
	if (!probes_.empty() && ImGui::BeginTable("probes", cols, tableFlags))
	{
		ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed, 22.0f);
		ImGui::TableSetupColumn("h (m)");
		ImGui::TableSetupColumn("|u| (m/s)");
		ImGui::TableSetupColumn("eta (m)");
		if (c.gw.enabled)
			ImGui::TableSetupColumn("nappe (m)");
		ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 22.0f);
		ImGui::TableHeadersRow();
		for (size_t p = 0; p < probes_.size(); ++p)
		{
			auto const &v = scalars_.probes[p];
			ImGui::TableNextRow();
			ImGui::TableNextColumn();
			ImGui::TextColored(probeColor(p), "%zu", p + 1);
			if (ImGui::IsItemHovered())
				ImGui::SetTooltip("maille (%u, %u) : x = %.1f m, y = %.1f m", probes_[p].i, probes_[p].j,
				                  (probes_[p].i + 0.5) * c.dx(), (probes_[p].j + 0.5) * c.dy());
			ImGui::TableNextColumn();
			ImGui::Text("%.3f", v.h);
			ImGui::TableNextColumn();
			ImGui::Text("%.3f", std::hypot(v.u, v.v));
			ImGui::TableNextColumn();
			ImGui::Text("%.3f", v.eta + c.z_ref);
			if (c.gw.enabled)
			{
				ImGui::TableNextColumn();
				ImGui::Text("%.3f", v.gwHead + c.z_ref);
			}
			ImGui::TableNextColumn();
			ImGui::PushID(int(p));
			if (ImGui::SmallButton("x"))
				remove = int(p);
			ImGui::PopID();
		}
		ImGui::EndTable();
	}
	if (remove >= 0)
	{
		probes_.erase(probes_.begin() + remove);
		updateProbeList();
	}
	if (!probes_.empty() && ImGui::SmallButton("Tout effacer"))
	{
		probes_.clear();
		updateProbeList();
	}
}

void App::drawDisplaySection()
{
	auto const &c = session_->caseData();
	ImGui::SetNextItemWidth(-1);
	if (ImGui::BeginCombo("##colour", colorModeInfo(render_.colorMode).label))
	{
		for (auto const &m : kColorModes)
			if (available(m, c) && ImGui::Selectable(m.label, m.mode == render_.colorMode))
			{
				render_.colorMode = m.mode;
				render_.cmapMax   = render_.cmapMin;        // the automatic range starts afresh
			}
		ImGui::EndCombo();
	}
	tip("Coloration de l'eau : rendu réaliste ou carte d'une grandeur.\n"
	    "Les grandeurs de la nappe colorent le sol, l'eau garde son rendu réaliste.");
	if (render_.colorMode != ColorMode::Realistic)
	{
		ImGui::Checkbox("Plage automatique", &autoRange_);
		ImGui::BeginDisabled(autoRange_);
		ImGui::SetNextItemWidth(-1);
		ImGui::DragFloatRange2("##range", &render_.cmapMin, &render_.cmapMax, 0.01f, -1e6f, 1e6f, "min %.3g", "max %.3g");
		ImGui::EndDisabled();
		ImGui::SetNextItemWidth(-90);
		ImGui::SliderFloat("opacité", &render_.cmapOpacity, 0.2f, 1.0f, "%.2f");
	}

	ImGui::SeparatorText("Éléments");
	ImGui::Checkbox("Eau", &render_.showWater);
	ImGui::SameLine(110);
	ImGui::Checkbox("Maillage", &render_.showMesh);
	ImGui::SameLine(220);
	ImGui::Checkbox("Zones", &render_.showZones);
	ImGui::Checkbox("Lignes de courant", &render_.showLines);
	tip("Germes dérivant avec l'écoulement, lignes tracées dans la direction de la vitesse");
	ImGui::SameLine(220);
	ImGui::Checkbox("Eau mince", &render_.thinWater);
	tip("Opacité minimale des lames d'eau de quelques cm");
	bool turbid = effects_->turbidity();
	if (ImGui::Checkbox("Eau turbide (crue chargée)", &turbid))
		effects_->setTurbidity(turbid);
	tip("L'eau rapide se charge en sédiments et devient brune, puis décante en eau calme");
	if (renderer_->hasOrthophoto())
	{
		ImGui::Checkbox("Orthophoto", &render_.showOrtho);
		if (render_.showOrtho)
		{
			ImGui::SameLine();
			ImGui::SetNextItemWidth(-90);
			ImGui::SliderFloat("mélange", &render_.orthoMix, 0.0f, 1.0f, "%.2f");
		}
	}

	if (c.gw.enabled)
	{
		ImGui::SeparatorText("Souterrain");
		ImGui::Checkbox("Nappe en transparence", &render_.gwXray);
		tip("Surface de la nappe vue à travers le sol (bleu clair : proche de la surface, foncé : profonde)");
		bool gwLines = effects_->groundwaterLines();
		if (ImGui::Checkbox("Lignes de courant de la nappe", &gwLines))
			effects_->setGroundwaterLines(gwLines);
		tip("Les lignes de courant suivent l'écoulement de la nappe (vitesse de pore -kh grad H / Sy), en bleu sur le sol");
		ImGui::Checkbox("Isopièzes", &render_.gwContours);
		tip("Courbes de niveau de la nappe dessinées sur le sol (une ligne foncée toutes les 5)");
		if (render_.gwContours)
		{
			ImGui::SameLine();
			ImGui::SetNextItemWidth(-90);
			ImGui::DragFloat("équidistance", &render_.gwContourStep, 0.005f, 0.005f, 50.0f, "%.3g m", ImGuiSliderFlags_Logarithmic);
		}
	}

	// cut box: a block diagram of part of the domain, its walls show the soil column (groundwater cases)
	ImGui::SeparatorText("Coupe");
	if (ImGui::Checkbox("Boîte de coupe", &render_.cropOn) && render_.cropOn && render_.crop.z <= render_.crop.x)
		render_.crop = {0, 0, c.nx - 1, c.ny / 2};        // first use: the southern half
	tip("N'affiche qu'un rectangle de mailles : ses parois montrent la colonne de sol\n"
	    "(zone non saturée, nappe en bleu, substratum gris) et l'eau");
	if (render_.cropOn)
	{
		int i0 = int(render_.crop.x), i1 = int(render_.crop.z), j0 = int(render_.crop.y), j1 = int(render_.crop.w);
		ImGui::SetNextItemWidth(-40);
		if (ImGui::DragIntRange2("i", &i0, &i1, 1.0f, 0, int(c.nx) - 1, "%d", "%d", ImGuiSliderFlags_AlwaysClamp))
			render_.crop.x = uint32_t(i0), render_.crop.z = uint32_t(i1);
		ImGui::SetNextItemWidth(-40);
		if (ImGui::DragIntRange2("j", &j0, &j1, 1.0f, 0, int(c.ny) - 1, "%d", "%d", ImGuiSliderFlags_AlwaysClamp))
			render_.crop.y = uint32_t(j0), render_.crop.w = uint32_t(j1);
		if (ImGui::SmallButton("Moitié sud"))
			render_.crop = {0, 0, c.nx - 1, c.ny / 2};
		ImGui::SameLine();
		if (ImGui::SmallButton("Moitié ouest"))
			render_.crop = {0, 0, c.nx / 2, c.ny - 1};
		ImGui::SameLine();
		if (ImGui::SmallButton("Tout"))
			render_.crop = {0, 0, c.nx - 1, c.ny - 1};
	}

	ImGui::SeparatorText("Vue");
	drawCameraButtons();
	ImGui::SetNextItemWidth(-90);
	ImGui::SliderFloat("exagération", &render_.z_scale, 1.0f, 2.5f * zScale0_, "x %.2f");
	tip("Exagération verticale (les ombres sont recalculées)");
}

void App::drawAdvancedSection()
{
	auto const &c = session_->caseData();
	ImGui::TextDisabled("MUSCL + Venkatakrishnan, HLLC, reconstruction hydrostatique");
	ImGui::SetNextItemWidth(-90);
	if (ImGui::BeginCombo("Runge-Kutta", schemes().at(c.scheme).label.c_str()))
	{
		for (auto const &[name, tableau] : schemes())
		{
			bool selected = name == c.scheme;
			if (ImGui::Selectable(std::format("{} (ordre {})", tableau.label, tableau.order).c_str(), selected) && !selected)
				session_->setScheme(name);
			if (selected)
				ImGui::SetItemDefaultFocus();
		}
		ImGui::EndCombo();
	}
	tip("Schémas SSP (combinaisons convexes de pas d'Euler) : h >= 0 est conservé exactement par le limiteur");
	int passes = c.drain_passes;
	ImGui::SetNextItemWidth(-90);
	if (ImGui::SliderInt("Drainage", &passes, 0, MAX_DRAIN_PASSES, passes == 0 ? "désactivé" : "%d passes", ImGuiSliderFlags_AlwaysClamp))
		session_->setDrainPasses(passes);
	tip("Limiteur de vidange : réduit les flux sortants des mailles qui se videraient (h >= 0 sans créer de masse).\n"
	    "Passes de Jacobi : 1 = seul le volume propre de la maille peut sortir, plus = l'eau reçue au même pas\n"
	    "est aussi admise. 0 = ancien comportement (h < 0 tronqué, masse créée).");
	float cfl = static_cast<float>(c.cfl);
	ImGui::SetNextItemWidth(-90);
	if (ImGui::SliderFloat("CFL", &cfl, 0.05f, 0.9f, "%.2f"))
		session_->setCfl(cfl);
	tip("dt = CFL x min(dx, dy) / vitesse d'onde max");
	float manning = static_cast<float>(c.manning_n);
	ImGui::SetNextItemWidth(-90);
	if (ImGui::SliderFloat("Manning n", &manning, 0.0f, 0.1f, "%.3f"))
		session_->setManning(manning);
	tip("Frottement de fond (s/m^1/3) : 0.03 lit naturel, 0.04 plaine, 0 sans frottement");
}

void App::drawSimulationPanel()
{
	auto const &c = session_->caseData();
	ImGui::TextWrapped("%s", c.label.c_str());
	ImGui::TextDisabled("%u x %u mailles de %.3g m  (%.2f M)", c.nx, c.ny, c.dx(), c.cells() * 1e-6);
	ImGui::Spacing();
	drawTransportSection();
	ImGui::Spacing();
	if (ImGui::CollapsingHeader("Temps et bilan", ImGuiTreeNodeFlags_DefaultOpen))
		drawBalanceSection();
	if (ImGui::CollapsingHeader(probes_.empty() ? "Sondes" : std::format("Sondes ({})###sondes", probes_.size()).c_str(),
	                            probes_.empty() ? 0 : ImGuiTreeNodeFlags_DefaultOpen))
		drawProbesSection();
	if (ImGui::CollapsingHeader("Affichage", ImGuiTreeNodeFlags_DefaultOpen))
		drawDisplaySection();
	if (ImGui::CollapsingHeader("Schéma numérique"))
		drawAdvancedSection();
}

// ---------------------------------------------------------------------------------------------
// Plots band (Simulation stage)
// ---------------------------------------------------------------------------------------------

void App::drawPlotsBand(ImVec2 pos, ImVec2 size)
{
	ImGui::SetNextWindowPos(pos);
	ImGui::SetNextWindowSize(size);
	ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
	if (!ImGui::Begin("##plots", nullptr, kFixedWindow))
	{
		ImGui::End();
		ImGui::PopStyleVar();
		return;
	}
	auto const &h = history_;
	int         n = static_cast<int>(h.t.size());
	auto const &c = session_->caseData();
	if (ImGui::BeginTabBar("plots"))
	{
		auto probePlot = [&](char const *tab, char const *ylabel, std::array<std::vector<double>, MAX_PROBES> const &series,
		                     double offset) {
			if (!ImGui::BeginTabItem(tab))
				return;
			if (probes_.empty())
				ImGui::TextDisabled("Aucune sonde : Ctrl + clic sur le terrain.");
			else if (ImPlot::BeginPlot(tab, ImVec2(-1, -1), ImPlotFlags_NoTitle))
			{
				ImPlot::SetupAxes("t (s)", ylabel, ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
				for (size_t p = 0; p < probes_.size(); ++p)
				{
					std::vector<double> y(series[p]);
					for (auto &v : y)
						v += offset;
					ImPlotSpec spec;
					spec.LineColor  = probeColor(p);
					spec.LineWeight = 2.0f;
					spec.Flags      = ImPlotLineFlags_SkipNaN;
					ImPlot::PlotLine(std::format("sonde {}", p + 1).c_str(), h.t.data(), y.data(), n, spec);
				}
				ImPlot::EndPlot();
			}
			ImGui::EndTabItem();
		};
		probePlot("Niveau", "eta (m)", h.eta, c.z_ref);
		probePlot("Hauteur", "h (m)", h.depth, 0.0);
		probePlot("Vitesse", "|u| (m/s)", h.speed, 0.0);
		if (c.gw.enabled)
			probePlot("Piézomètres", "cote de la nappe (m)", h.gwHead, c.z_ref);
		bool chronologies = c.infiltration.active() || c.rain.active();
		if ((c.gw.enabled || chronologies) && ImGui::BeginTabItem(c.gw.enabled ? "Nappe" : "Pluie"))
		{
			if (ImPlot::BeginPlot("vertical", ImVec2(-1, -1), ImPlotFlags_NoTitle))
			{
				ImPlot::SetupAxes("t (s)", "volume (m3)", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
				ImPlot::PlotLine("surface", h.t.data(), h.volume.data(), n);
				if (c.gw.enabled)
				{
					ImPlot::PlotLine("nappe", h.t.data(), h.gwVolume.data(), n);
					ImPlot::PlotLine("infiltré", h.t.data(), h.gwInfil.data(), n);
					ImPlot::PlotLine("exfiltré", h.t.data(), h.gwExfil.data(), n);
				}
				if (chronologies)
					ImPlot::PlotLine("pluie cumulée", h.t.data(), h.rain.data(), n);
				ImPlot::EndPlot();
			}
			ImGui::EndTabItem();
		}
		if (ImGui::BeginTabItem("Volume"))
		{
			if (ImPlot::BeginPlot("volume", ImVec2(-1, -1), ImPlotFlags_NoTitle))
			{
				ImPlot::SetupAxes("t (s)", "volume (m3)", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
				ImPlot::SetupAxis(ImAxis_Y2, "emprise (m2)", ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_Opposite);
				ImPlot::PlotLine("volume", h.t.data(), h.volume.data(), n);
				ImPlot::SetAxes(ImAxis_X1, ImAxis_Y2);
				ImPlot::PlotLine("emprise mouillée", h.t.data(), h.wetArea.data(), n);
				ImPlot::EndPlot();
			}
			ImGui::EndTabItem();
		}
		if (ImGui::BeginTabItem("Profil", nullptr, std::exchange(selectProfileTab_, false) ? ImGuiTabItemFlags_SetSelected : 0))
		{
			drawProfilePlot();
			ImGui::EndTabItem();
		}
		if (ImGui::BeginTabItem("Pas de temps"))
		{
			if (ImPlot::BeginPlot("dt", ImVec2(-1, -1), ImPlotFlags_NoTitle))
			{
				ImPlot::SetupAxes("t (s)", "dt (s)", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
				ImPlot::PlotLine("dt", h.t.data(), h.dt.data(), n);
				ImPlot::EndPlot();
			}
			ImGui::EndTabItem();
		}
		// collapse button at the right end of the tab bar
		if (ImGui::TabItemButton("  v  ", ImGuiTabItemFlags_Trailing | ImGuiTabItemFlags_NoTooltip))
			showPlots_ = false;
		ImGui::EndTabBar();
	}
	ImGui::End();
	ImGui::PopStyleVar();
}

// Cells of the profile for the current mode; sent to the solver when they change
void App::updateProfileCells()
{
	auto const           &c = session_->caseData();
	std::vector<uint32_t> cells;
	if (profileMode_ == 2 && probes_.size() >= 2)
	{
		// probe 1 -> probe 2, one cell per step along the longer axis
		int  i0 = int(probes_[0].i), j0 = int(probes_[0].j), i1 = int(probes_[1].i), j1 = int(probes_[1].j);
		int  n  = std::max(std::abs(i1 - i0), std::abs(j1 - j0)) + 1;
		for (int k = 0; k < n && cells.size() < MAX_PROFILE; ++k)
		{
			double t = n > 1 ? double(k) / (n - 1) : 0.0;
			cells.push_back(uint32_t(std::lround(i0 + t * (i1 - i0))) * c.ny + uint32_t(std::lround(j0 + t * (j1 - j0))));
		}
	}
	else if (profileMode_ == 1)
	{
		uint32_t i = profileIndex_ < 0 ? c.nx / 2 : std::min(uint32_t(profileIndex_), c.nx - 1);
		for (uint32_t j = 0; j < c.ny && cells.size() < MAX_PROFILE; ++j)
			cells.push_back(i * c.ny + j);
	}
	else if (profileMode_ == 0)
	{
		uint32_t j = profileIndex_ < 0 ? c.ny / 2 : std::min(uint32_t(profileIndex_), c.ny - 1);
		for (uint32_t i = 0; i < c.nx && cells.size() < MAX_PROFILE; ++i)
			cells.push_back(i * c.ny + j);
	}
	if (cells != profileCells_)
	{
		profileCells_ = std::move(cells);
		session_->solver().setProfile(profileCells_);
		profile_.clear();        // the next readback carries the new line
	}
}

// Vertical section along a line: substratum, saturated soil (aquifer), unsaturated soil, surface water
void App::drawProfilePlot()
{
	auto const &c = session_->caseData();
	ImGui::SetNextItemWidth(220);
	ImGui::Combo("##pmode", &profileMode_, "le long de x (ligne j)\0le long de y (colonne i)\0sonde 1 -> sonde 2\0");
	if (profileMode_ < 2)
	{
		int n   = int(profileMode_ == 0 ? c.ny : c.nx);
		int idx = profileIndex_ < 0 ? n / 2 : std::min(profileIndex_, n - 1);
		ImGui::SameLine();
		ImGui::SetNextItemWidth(-1);
		if (ImGui::SliderInt("##pidx", &idx, 0, n - 1, profileMode_ == 0 ? "j = %d" : "i = %d"))
			profileIndex_ = idx;
	}
	else if (probes_.size() < 2)
	{
		ImGui::SameLine();
		ImGui::TextDisabled("poser deux sondes (Ctrl + clic)");
	}
	updateProfileCells();
	size_t const n = std::min(profile_.size(), profileCells_.size());
	if (n < 2)
		return;

	// abscissa: distance along the line (m); elevations in m DNG
	std::vector<double> x(n), zb(n), eta(n), head(n), zsub(n), floor(n);
	double const        zr = c.z_ref;
	double              lo = 1e30;
	for (size_t k = 0; k < n; ++k)
	{
		uint32_t a = profileCells_[0], b = profileCells_[k];
		double   di = (double(b / c.ny) - double(a / c.ny)) * c.dx(), dj = (double(b % c.ny) - double(a % c.ny)) * c.dy();
		x[k]    = std::sqrt(di * di + dj * dj);
		zb[k]   = profile_[k].zb + zr;
		eta[k]  = zb[k] + profile_[k].h;
		head[k] = std::min(profile_[k].head, profile_[k].zb) + zr;
		zsub[k] = profile_[k].zSub + zr;
		lo      = std::min(lo, c.gw.enabled ? zsub[k] : zb[k]);
	}
	double depth = std::max(c.depth_ref, 0.1);
	std::fill(floor.begin(), floor.end(), lo - 0.3 * depth);
	if (ImPlot::BeginPlot("profil", ImVec2(-1, -1), ImPlotFlags_NoTitle))
	{
		ImPlot::SetupAxes("distance (m)", "cote (m)", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
		auto fill = [&](char const *label, std::vector<double> const &y0, std::vector<double> const &y1, ImVec4 col) {
			ImPlotSpec spec;
			spec.FillColor = col;
			ImPlot::PlotShaded(label, x.data(), y0.data(), y1.data(), int(n), spec);
		};
		auto line = [&](char const *label, std::vector<double> const &y, ImVec4 col, float w) {
			ImPlotSpec spec;
			spec.LineColor  = col;
			spec.LineWeight = w;
			ImPlot::PlotLine(label, x.data(), y.data(), int(n), spec);
		};
		if (c.gw.enabled)
		{
			fill("substratum", floor, zsub, ImVec4(0.45f, 0.45f, 0.47f, 1.0f));
			fill("sol saturé (nappe)", zsub, head, ImVec4(0.25f, 0.42f, 0.75f, 1.0f));
			fill("sol non saturé", head, zb, ImVec4(0.55f, 0.40f, 0.25f, 1.0f));
		}
		else
			fill("sol", floor, zb, ImVec4(0.55f, 0.40f, 0.25f, 1.0f));
		fill("eau", zb, eta, ImVec4(0.35f, 0.7f, 1.0f, 0.8f));
		line("terrain", zb, ImVec4(0.15f, 0.1f, 0.05f, 1.0f), 2.0f);
		if (c.gw.enabled)
			line("cote de la nappe", head, ImVec4(0.0f, 0.15f, 0.6f, 1.0f), 2.0f);
		ImPlot::EndPlot();
	}
}

// ---------------------------------------------------------------------------------------------
// Floating windows and overlays
// ---------------------------------------------------------------------------------------------

void App::drawPerfWindow()
{
	ImGui::SetNextWindowSize(ImVec2(360, 0), ImGuiCond_FirstUseEver);
	if (!ImGui::Begin("Performances", &showPerf_))
	{
		ImGui::End();
		return;
	}
	auto const &io = ImGui::GetIO();
	auto const &c  = session_->caseData();
	ImGui::TextUnformatted(ctx_->properties.deviceName.data());
	ImGui::Text("%.1f images/s   (%.2f ms)", io.Framerate, 1000.0f / io.Framerate);
	ImGui::Text("%.0f pas/s   %.0f M mailles.pas/s", stepsPerSec_, stepsPerSec_ * c.cells() * 1e-6);
	if (timestampPeriod_ > 0.0)
	{
		ImGui::SeparatorText("GPU par image");
		double perStep = stepsPerFrame_ > 0 && !paused_ ? gpu_.solver / stepsPerFrame_ * 1000.0 : 0.0;
		ImGui::Text("solveur   %7.3f ms  (%.1f us/pas)", gpu_.solver, perStep);
		ImGui::Text("effets    %7.3f ms", gpu_.effects);
		ImGui::Text("rendu     %7.3f ms", gpu_.render);
		tip("Rendu 3D + interface, MSAA 4x");
	}
	else
		ImGui::TextDisabled("Timestamps GPU non disponibles");
	ImGui::SeparatorText("Rendu");
	ImGui::Text("vue %.0f x %.0f px, MSAA %ux", viewRect_.z, viewRect_.w, static_cast<unsigned>(renderer_->samples()));
	ImGui::End();
}

// Colour bar in the bottom-right corner of the 3D view
void App::drawLegend()
{
	if (render_.colorMode == ColorMode::Realistic)
		return;
	auto const &info = colorModeInfo(render_.colorMode);
	if (!available(info, session_->caseData()))
		return;
	auto   cmap  = render_.colorMode == ColorMode::GwExchange ? diverging : turbo;
	ImVec2 scale = ImGui::GetIO().DisplayFramebufferScale;
	auto  *vp    = ImGui::GetMainViewport();
	ImVec2 corner(vp->Pos.x + (viewRect_.x + viewRect_.z) / scale.x - 16.0f, vp->Pos.y + (viewRect_.y + viewRect_.w) / scale.y - 16.0f);
	ImGui::SetNextWindowPos(corner, ImGuiCond_Always, ImVec2(1.0f, 1.0f));
	ImGui::SetNextWindowBgAlpha(0.75f);
	ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings |
	                         ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoMove;
	if (ImGui::Begin("##legend", nullptr, flags))
	{
		if (*info.unit)
			ImGui::Text("%s (%s)", info.label, info.unit);
		else
			ImGui::TextUnformatted(info.label);
		ImVec2    p0 = ImGui::GetCursorScreenPos();
		float     w = 240.0f, hbar = 14.0f;
		auto     *dl    = ImGui::GetWindowDrawList();
		int const steps = 48;
		for (int k = 0; k < steps; ++k)
		{
			glm::vec3 a = cmap(float(k) / steps), b = cmap(float(k + 1) / steps);
			dl->AddRectFilledMultiColor(ImVec2(p0.x + w * k / steps, p0.y), ImVec2(p0.x + w * (k + 1) / steps, p0.y + hbar),
			                            ImGui::GetColorU32(ImVec4(a.r, a.g, a.b, 1)), ImGui::GetColorU32(ImVec4(b.r, b.g, b.b, 1)),
			                            ImGui::GetColorU32(ImVec4(b.r, b.g, b.b, 1)), ImGui::GetColorU32(ImVec4(a.r, a.g, a.b, 1)));
		}
		ImGui::Dummy(ImVec2(w, hbar));
		bool   absolute = render_.colorMode == ColorMode::Elevation || render_.colorMode == ColorMode::GwHead;
		double offset   = absolute ? session_->caseData().z_ref : 0.0;
		auto   label    = [&](float v) { return std::format("{:.3g}", v + offset); };
		std::string lo = label(render_.cmapMin), mid = label(0.5f * (render_.cmapMin + render_.cmapMax)), hi = label(render_.cmapMax);
		ImGui::TextUnformatted(lo.c_str());
		ImGui::SameLine(w * 0.5f - ImGui::CalcTextSize(mid.c_str()).x * 0.5f + ImGui::GetStyle().WindowPadding.x);
		ImGui::TextUnformatted(mid.c_str());
		ImGui::SameLine(w - ImGui::CalcTextSize(hi.c_str()).x + ImGui::GetStyle().WindowPadding.x);
		ImGui::TextUnformatted(hi.c_str());
	}
	ImGui::End();
}

// Numbered markers of the probes, drawn over the 3D view (under the panels)
void App::drawProbeMarkers()
{
	if (probes_.empty())
		return;
	auto const &c     = session_->caseData();
	float       S     = renderer_->S();
	ImVec2      scale = ImGui::GetIO().DisplayFramebufferScale;
	auto       *vp    = ImGui::GetMainViewport();
	glm::mat4   vpm   = camera_.proj(viewRect_.z / std::max(viewRect_.w, 1.0f)) * camera_.view();
	auto       *dl    = ImGui::GetBackgroundDrawList();
	dl->PushClipRect(ImVec2(vp->Pos.x + viewRect_.x / scale.x, vp->Pos.y + viewRect_.y / scale.y),
	                 ImVec2(vp->Pos.x + (viewRect_.x + viewRect_.z) / scale.x, vp->Pos.y + (viewRect_.y + viewRect_.w) / scale.y), true);
	for (size_t p = 0; p < probes_.size(); ++p)
	{
		auto  k   = size_t(probes_[p].i) * c.ny + probes_[p].j;
		float eta = scalars_.probes[p].eta != 0.0f ? scalars_.probes[p].eta : c.zb[k];
		glm::vec4 clip = vpm * glm::vec4((probes_[p].i + 0.5f) * float(c.dx()) * S, eta * render_.z_scale * S,
		                                 -(probes_[p].j + 0.5f) * float(c.dy()) * S, 1.0f);
		if (clip.w <= 0.0f)
			continue;
		glm::vec2 ndc(clip.x / clip.w, clip.y / clip.w);
		ImVec2    pos(vp->Pos.x + (viewRect_.x + (ndc.x * 0.5f + 0.5f) * viewRect_.z) / scale.x,
		              vp->Pos.y + (viewRect_.y + (ndc.y * 0.5f + 0.5f) * viewRect_.w) / scale.y);
		ImU32     col = ImGui::GetColorU32(probeColor(p));
		dl->AddLine(pos, ImVec2(pos.x, pos.y - 22.0f), col, 2.0f);
		dl->AddCircleFilled(ImVec2(pos.x, pos.y - 30.0f), 9.0f, col);
		auto label = std::to_string(p + 1);
		auto ts    = ImGui::CalcTextSize(label.c_str());
		dl->AddText(ImVec2(pos.x - ts.x * 0.5f, pos.y - 30.0f - ts.y * 0.5f), IM_COL32(20, 20, 20, 255), label.c_str());
		dl->AddCircleFilled(pos, 3.0f, col);
	}
	dl->PopClipRect();
}

void App::drawHelp()
{
	ImGui::SetNextWindowSize(ImVec2(460, 0), ImGuiCond_FirstUseEver);
	if (ImGui::Begin("Aide", &showHelp_))
	{
		auto row = [](char const *key, char const *what) {
			ImGui::TextColored(ImVec4(0.6f, 0.8f, 1.0f, 1.0f), "%-18s", key);
			ImGui::SameLine(150);
			ImGui::TextUnformatted(what);
		};
		ImGui::SeparatorText("Étapes");
		ImGui::TextWrapped("1 Carte : tracer une zone et importer le relief, ou ouvrir un cas. 2 Terrain : vérifier le relief. "
		                   "3 Conditions : eau, sol, pluie, zones et bords sur le maillage, à t = 0. 4 Simulation : lancer et "
		                   "observer. La barre du haut permet de revenir à une étape précédente.");
		ImGui::SeparatorText("Souris (vue 3D)");
		row("Clic gauche", "tourner (étape Conditions : sélectionner des mailles)");
		row("Clic milieu", "déplacer");
		row("Molette", "zoomer");
		row("Ctrl + clic", "poser une sonde");
		ImGui::SeparatorText("Carte");
		row("Glisser", "déplacer la carte");
		row("Clic droit + glisser", "tracer la zone à importer");
		ImGui::SeparatorText("Clavier");
		row("Espace", "pause / reprise (étape Simulation)");
		row("R", "recommencer (étape Simulation)");
		row("Échap", "annuler la sélection");
		row("Origine", "recadrer la caméra");
		row("F12", "capture d'écran (captures/)");
		row("F11", "masquer / afficher l'interface");
		row("F1", "cette aide");
	}
	ImGui::End();
}

// ---------------------------------------------------------------------------------------------
// Probes and picking
// ---------------------------------------------------------------------------------------------

// Ray from the camera through the mouse, marched against the (exaggerated) bed surface.
std::optional<App::Probe> App::pickCell(glm::vec2 mouse) const
{
	auto const &c     = session_->caseData();
	ImVec2      scale = ImGui::GetIO().DisplayFramebufferScale;
	auto       *vp    = ImGui::GetMainViewport();
	glm::vec2   px((mouse.x - vp->Pos.x) * scale.x, (mouse.y - vp->Pos.y) * scale.y);
	glm::vec2   ndc(2.0f * (px.x - viewRect_.x) / viewRect_.z - 1.0f, 2.0f * (px.y - viewRect_.y) / viewRect_.w - 1.0f);
	if (std::abs(ndc.x) > 1.0f || std::abs(ndc.y) > 1.0f)
		return std::nullopt;

	glm::mat4 inv  = glm::inverse(camera_.proj(viewRect_.z / std::max(viewRect_.w, 1.0f)) * camera_.view());
	glm::vec4 a    = inv * glm::vec4(ndc, 0.0f, 1.0f);
	glm::vec4 b    = inv * glm::vec4(ndc, 1.0f, 1.0f);
	glm::vec3 from = glm::vec3(a) / a.w, to = glm::vec3(b) / b.w;
	glm::vec3 dir  = glm::normalize(to - from);

	float  S    = renderer_->S();
	double dx   = c.dx(), dy = c.dy();
	float  zs   = render_.z_scale;
	auto   bedAt = [&](float X, float Z, uint32_t &i, uint32_t &j) -> std::optional<float> {
		double x = X / S, y = -Z / S;
		if (x < 0 || y < 0 || x >= c.Lx || y >= c.Ly)
			return std::nullopt;
		i = std::min(uint32_t(x / dx), c.nx - 1);
		j = std::min(uint32_t(y / dy), c.ny - 1);
		return c.zb[size_t(i) * c.ny + j] * zs * S;
	};
	float step = 0.25f * float(std::min(dx, dy)) * S;
	float maxT = glm::length(to - from);
	bool  wasInside = false;
	for (float t = 0.0f; t < maxT; t += step)
	{
		glm::vec3 p = from + dir * t;
		uint32_t  i, j;
		auto      bed = bedAt(p.x, p.z, i, j);
		if (!bed)
		{
			if (wasInside)
				break;        // left the domain
			continue;
		}
		wasInside = true;
		if (p.y <= *bed)
			return Probe{i, j};
	}
	return std::nullopt;
}

void App::updateProbeList()
{
	auto const           &c = session_->caseData();
	std::vector<uint32_t> cells;
	for (auto const &p : probes_)
		cells.push_back(p.i * c.ny + p.j);
	session_->solver().setProbes(cells);
	history_.clearProbes();
	for (auto &v : scalars_.probes)
		v = {};
}

// Colour map range following the current maxima (smoothed to avoid flicker)
void App::autoColorRange()
{
	auto const &c = session_->caseData();
	float       lo = 0.0f, hi = 1.0f;
	switch (render_.colorMode)
	{
	case ColorMode::Depth: hi = std::max(scalars_.hMax, 1e-3f); break;
	case ColorMode::Speed: hi = std::max(scalars_.uMax, 1e-3f); break;
	case ColorMode::Froude: hi = 2.0f; break;
	case ColorMode::Elevation:
	{
		auto [mn, mx] = session_->bedRange();        // cached: a scan of zb every frame costs ms on large cases
		lo            = mn;
		hi            = std::max(mn + scalars_.hMax, static_cast<float>(std::min<double>(c.eta_0, mx)));
		break;
	}
	// groundwater: bounds from the soil (no per-cell maxima are reduced on the GPU)
	case ColorMode::GwTableDepth: hi = session_->gwStats().thicknessMax; break;
	case ColorMode::GwSaturation: hi = 1.0f; break;
	case ColorMode::GwExchange:
		hi = std::max(session_->gwStats().kvMax * 3.6e6f, 1e-6f);        // mm/h, symmetric: white = no exchange
		lo = -hi;
		break;
	case ColorMode::GwInfiltrated:
		// the net infiltration of a cell is at most kv t; follow the mean depth so the map does not saturate
		hi = std::max(1e3f * std::min(session_->gwStats().kvMax * float(scalars_.t),
		                              4.0f * scalars_.gwInfil / float(c.Lx * c.Ly)),
		              1e-3f);
		break;
	case ColorMode::GwHead:
		lo = session_->gwStats().zSubMin;
		hi = session_->bedRange().second;
		break;
	case ColorMode::Rain:        // twice the mean depth: zones with more rain than average still read
		hi = std::max(2e3f * scalars_.srcRain / float(c.Lx * c.Ly), 1e-3f);
		break;
	default: return;
	}
	if (!std::isfinite(lo) || !std::isfinite(hi))
		return;
	bool first      = render_.cmapMax <= render_.cmapMin;
	render_.cmapMin = first ? lo : render_.cmapMin + 0.1f * (lo - render_.cmapMin);
	render_.cmapMax = first ? hi : render_.cmapMax + 0.1f * (hi - render_.cmapMax);
}

}        // namespace swe
