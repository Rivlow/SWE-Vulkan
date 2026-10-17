// GUI panels of the application (ImGui docking + ImPlot). The frame loop is in App.cpp.
#include "app/App.h"

#include "util/Log.h"

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_internal.h>
#include <implot.h>

#include <glm/gtc/matrix_inverse.hpp>

#include <algorithm>
#include <cmath>

namespace swe {

namespace {

char const *kSimulation = "Simulation";
char const *kScheme     = "Schéma numérique";
char const *kDisplay    = "Affichage";
char const *kProbes     = "Sondes";
char const *kPlots      = "Courbes";
char const *kPerf       = "Performances";

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
	if (!std::isfinite(t))
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
};
constexpr ColorModeInfo kColorModes[] = {
    {ColorMode::Realistic, "Réaliste", ""},
    {ColorMode::Depth, "Hauteur d'eau h", "m"},
    {ColorMode::Speed, "Vitesse |u|", "m/s"},
    {ColorMode::Froude, "Nombre de Froude", ""},
    {ColorMode::Elevation, "Cote de la surface libre", "m"},
};

ColorModeInfo const &colorModeInfo(ColorMode m)
{
	return kColorModes[static_cast<int>(m)];
}

}        // namespace

// ---------------------------------------------------------------------------------------------
// Frame GUI
// ---------------------------------------------------------------------------------------------

void App::drawGui()
{
	drawMenuBar();

	ImGuiID dock = ImGui::DockSpaceOverViewport(0, nullptr, ImGuiDockNodeFlags_PassthruCentralNode);
	setupDockLayout(dock);

	// 3D view = central node of the dock space (in framebuffer pixels)
	if (ImGuiDockNode *central = ImGui::DockBuilderGetCentralNode(dock))
	{
		auto  *vp    = ImGui::GetMainViewport();
		ImVec2 scale = ImGui::GetIO().DisplayFramebufferScale;
		viewRect_    = glm::vec4((central->Pos.x - vp->Pos.x) * scale.x, (central->Pos.y - vp->Pos.y) * scale.y,
		                         central->Size.x * scale.x, central->Size.y * scale.y);
	}

	if (autoRange_)
		autoColorRange();

	if (showSimulation_)
		drawSimulationPanel();
	if (showScheme_)
		drawSchemePanel();
	if (showDisplay_)
		drawDisplayPanel();
	if (showProbes_)
		drawProbesPanel();
	if (showPlots_)
		drawPlotsPanel();
	if (showPerf_)
		drawPerfPanel();
	drawLegend();
	drawProbeMarkers();
	if (showHelp_)
		drawHelp();
	if (showDemo_)
		ImGui::ShowDemoWindow(&showDemo_);
}

// Default layout on first launch (afterwards imgui.ini keeps whatever the user arranged)
void App::setupDockLayout(unsigned int dockId)
{
	static bool checked = false;
	if (checked)
		return;
	checked = true;
	if (ImGui::FindWindowSettingsByID(ImHashStr(kSimulation)) != nullptr)
		return;

	ImGuiID dock = dockId;
	auto   *vp   = ImGui::GetMainViewport();
	ImGui::DockBuilderRemoveNode(dock);
	ImGui::DockBuilderAddNode(dock, ImGuiDockNodeFlags(ImGuiDockNodeFlags_DockSpace) | ImGuiDockNodeFlags_PassthruCentralNode);
	ImGui::DockBuilderSetNodeSize(dock, vp->WorkSize);

	ImGuiID center = dock;
	ImGuiID left   = ImGui::DockBuilderSplitNode(center, ImGuiDir_Left, 0.23f, nullptr, &center);
	ImGuiID right  = ImGui::DockBuilderSplitNode(center, ImGuiDir_Right, 0.24f, nullptr, &center);
	ImGuiID bottom = ImGui::DockBuilderSplitNode(center, ImGuiDir_Down, 0.30f, nullptr, &center);
	ImGuiID leftBottom;
	ImGuiID leftTop = ImGui::DockBuilderSplitNode(left, ImGuiDir_Up, 0.42f, nullptr, &leftBottom);

	ImGui::DockBuilderDockWindow(kSimulation, leftTop);
	ImGui::DockBuilderDockWindow(kDisplay, leftBottom);
	ImGui::DockBuilderDockWindow(kScheme, leftBottom);
	ImGui::DockBuilderDockWindow(kProbes, right);
	ImGui::DockBuilderDockWindow(kPerf, right);
	ImGui::DockBuilderDockWindow(kPlots, bottom);
	ImGui::DockBuilderFinish(dock);
}

void App::drawMenuBar()
{
	if (!ImGui::BeginMainMenuBar())
		return;
	if (ImGui::BeginMenu("Fichier"))
	{
		if (ImGui::MenuItem("Capture d'écran", "F12"))
			captureRequest_ = std::filesystem::path("captures") / std::format("swe_{:%Y%m%d_%H%M%S}.png",
			                                                                std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now()));
		if (ImGui::MenuItem(recording_ ? "Arrêter l'enregistrement" : "Enregistrer une séquence d'images"))
		{
			recording_ = !recording_;
			if (recording_)
			{
				recordDir_   = std::filesystem::path("captures") /
				             std::format("rec_{:%Y%m%d_%H%M%S}", std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now()));
				recordIndex_ = 0;
				std::filesystem::create_directories(recordDir_);
				log::info("recording to {}", std::filesystem::absolute(recordDir_).string());
			}
		}
		ImGui::Separator();
		if (ImGui::MenuItem("Quitter", "Alt+F4"))
			glfwSetWindowShouldClose(window_, GLFW_TRUE);
		ImGui::EndMenu();
	}
	if (ImGui::BeginMenu("Simulation"))
	{
		ImGui::MenuItem("Pause", "Espace", &paused_);
		if (ImGui::MenuItem("Recommencer", "R"))
			resetSimulation();
		if (ImGui::BeginMenu("Cas"))
		{
			for (auto const &info : caseList_)
				if (ImGui::MenuItem(info.label.c_str(), nullptr, info.name == solver_->caseData().name) &&
				    info.name != solver_->caseData().name)
					loadCase(info.name);
			ImGui::EndMenu();
		}
		ImGui::EndMenu();
	}
	if (ImGui::BeginMenu("Fenêtres"))
	{
		ImGui::MenuItem(kSimulation, nullptr, &showSimulation_);
		ImGui::MenuItem(kScheme, nullptr, &showScheme_);
		ImGui::MenuItem(kDisplay, nullptr, &showDisplay_);
		ImGui::MenuItem(kProbes, nullptr, &showProbes_);
		ImGui::MenuItem(kPlots, nullptr, &showPlots_);
		ImGui::MenuItem(kPerf, nullptr, &showPerf_);
		ImGui::Separator();
		if (ImGui::MenuItem("Recadrer la caméra", "Origine"))
			camera_.reset();
		ImGui::MenuItem("Démo ImGui", nullptr, &showDemo_);
		ImGui::EndMenu();
	}
	if (ImGui::BeginMenu("Aide"))
	{
		ImGui::MenuItem("Raccourcis et souris", "F1", &showHelp_);
		ImGui::EndMenu();
	}

	// status on the right of the menu bar
	auto const &c      = solver_->caseData();
	std::string status = std::format("{}   t = {}   {}{}", c.label, formatTime(scalars_.t), paused_ ? "EN PAUSE" : "",
	                                 recording_ ? "   ● REC" : "");
	float w = ImGui::CalcTextSize(status.c_str()).x;
	ImGui::SetCursorPosX(ImGui::GetWindowWidth() - w - 16.0f);
	ImGui::TextUnformatted(status.c_str());
	ImGui::EndMainMenuBar();
}

void App::drawSimulationPanel()
{
	if (!ImGui::Begin(kSimulation, &showSimulation_))
	{
		ImGui::End();
		return;
	}
	auto const &c = solver_->caseData();

	// case
	ImGui::SetNextItemWidth(-1);
	if (ImGui::BeginCombo("##case", c.label.c_str()))
	{
		for (auto const &info : caseList_)
			if (ImGui::Selectable(info.label.c_str(), info.name == c.name) && info.name != c.name)
				loadCase(info.name);
		ImGui::EndCombo();
	}
	tip("Cas de calcul (fichiers assets/cases/*.json)");
	ImGui::TextDisabled("%u x %u mailles de %.3g m  (%.2f M)", c.nx, c.ny, c.dx(), c.cells() * 1e-6);

	// transport
	ImGui::Spacing();
	float bw = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) * 0.5f;
	if (ImGui::Button(paused_ ? "Reprendre" : "Pause", ImVec2(bw, 0)))
		paused_ = !paused_;
	tip("Espace");
	ImGui::SameLine();
	if (ImGui::Button("Recommencer", ImVec2(bw, 0)))
		resetSimulation();
	tip("R : remet l'état initial du cas");
	ImGui::SetNextItemWidth(-90);
	ImGui::SliderInt("pas / image", &stepsPerFrame_, 1, 500, "%d", ImGuiSliderFlags_AlwaysClamp | ImGuiSliderFlags_Logarithmic);
	tip("Nombre de pas de temps calculés entre deux images : plus = simulation plus rapide, image moins fluide");

	// clock
	ImGui::SeparatorText("Temps");
	ImGui::Text("t  = %s   (%u pas)", formatTime(scalars_.t).c_str(), scalars_.n_steps);
	ImGui::Text("dt = %.3e s   s_max = %.3f m/s", scalars_.dt, scalars_.s_max);
	ImGui::Text("%.0f pas/s", stepsPerSec_);
	if (!std::isfinite(scalars_.t) || !std::isfinite(scalars_.dt) || !std::isfinite(scalars_.volume))
		ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.3f, 1.0f), "La simulation a divergé : Recommencer");

	// balance
	ImGui::SeparatorText("Bilan");
	double dV = volume0_ > 0.0 ? (scalars_.volume - volume0_) / volume0_ * 100.0 : 0.0;
	ImGui::Text("Volume     %s", formatVolume(scalars_.volume).c_str());
	ImGui::Text("           %+.4f %% depuis t = 0", dV);
	tip("Variation de volume : 0 avec des murs partout (conservation), sinon débits entrants / sortants");
	ImGui::Text("Emprise    %s", formatArea(scalars_.wetCells * c.dx() * c.dy()).c_str());
	ImGui::Text("h max      %.3f m", scalars_.hMax);
	ImGui::Text("|u| max    %.3f m/s", scalars_.uMax);
	ImGui::End();
}

void App::drawSchemePanel()
{
	if (!ImGui::Begin(kScheme, &showScheme_))
	{
		ImGui::End();
		return;
	}
	auto const &c = solver_->caseData();
	ImGui::TextDisabled("MUSCL + Venkatakrishnan, HLLC, reconstruction hydrostatique");

	ImGui::SetNextItemWidth(-90);
	if (ImGui::BeginCombo("Runge-Kutta", schemes().at(solver_->schemeName()).label.c_str()))
	{
		for (auto const &[name, tableau] : schemes())
		{
			bool selected = name == solver_->schemeName();
			if (ImGui::Selectable(std::format("{} (ordre {})", tableau.label, tableau.order).c_str(), selected) && !selected)
				solver_->setScheme(name);
			if (selected)
				ImGui::SetItemDefaultFocus();
		}
		ImGui::EndCombo();
	}
	tip("Les schémas SSP gardent h >= 0 : à privilégier avec des fronts secs");
	float cfl = static_cast<float>(c.cfl);
	ImGui::SetNextItemWidth(-90);
	if (ImGui::SliderFloat("CFL", &cfl, 0.05f, 0.9f, "%.2f"))
		solver_->setCfl(cfl);
	tip("dt = CFL x min(dx, dy) / vitesse d'onde max");
	float manning = static_cast<float>(c.manning_n);
	ImGui::SetNextItemWidth(-90);
	if (ImGui::SliderFloat("Manning n", &manning, 0.0f, 0.1f, "%.3f"))
		solver_->setManning(manning);
	tip("Frottement de fond (s/m^1/3) : 0.03 lit naturel, 0.04 plaine, 0 sans frottement");

	ImGui::SeparatorText("Conditions aux limites");
	static char const *sideNames[4] = {"Ouest", "Est", "Sud", "Nord"};
	static char const *kindNames[3] = {"mur", "débit entrant", "niveau imposé"};
	for (int s = 0; s < 4; ++s)
	{
		ImGui::PushID(s);
		auto bc   = c.bc[s];
		int  kind = static_cast<int>(bc.kind);
		ImGui::SetNextItemWidth(125);
		bool changed = ImGui::Combo(sideNames[s], &kind, kindNames, 3);
		if (changed && kind != static_cast<int>(bc.kind))
		{
			bc.kind  = static_cast<BcKind>(kind);
			bc.value = bc.kind == BcKind::Level ? c.eta_0 : bc.kind == BcKind::Inflow ? 10.0 : 0.0;
		}
		if (bc.kind != BcKind::Wall)
		{
			ImGui::SameLine();
			ImGui::SetNextItemWidth(-1);
			double v = bc.value;
			if (ImGui::InputDouble("##value", &v, 0.0, 0.0, bc.kind == BcKind::Inflow ? "%.1f m3/s" : "%.2f m",
			                       ImGuiInputTextFlags_EnterReturnsTrue))
			{
				bc.value = v;
				changed  = true;
			}
			tip(bc.kind == BcKind::Inflow ? "Débit total réparti sur le bord au prorata de h^(5/3) (Entrée pour valider)"
			                              : "Cote de surface libre imposée, relative à z_ref (Entrée pour valider)");
		}
		if (changed)
			solver_->setBoundary(s, bc);
		ImGui::PopID();
	}
	ImGui::TextDisabled("Cotes relatives à z_ref = %.0f m", c.z_ref);
	ImGui::End();
}

void App::drawDisplayPanel()
{
	if (!ImGui::Begin(kDisplay, &showDisplay_))
	{
		ImGui::End();
		return;
	}
	// colouring
	ImGui::SetNextItemWidth(-1);
	if (ImGui::BeginCombo("##colour", colorModeInfo(render_.colorMode).label))
	{
		for (auto const &m : kColorModes)
			if (ImGui::Selectable(m.label, m.mode == render_.colorMode))
				render_.colorMode = m.mode;
		ImGui::EndCombo();
	}
	tip("Coloration de l'eau : rendu réaliste ou carte d'une grandeur");
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
	ImGui::SameLine(120);
	ImGui::Checkbox("Maillage", &render_.showMesh);
	ImGui::Checkbox("Lignes de courant", &render_.showLines);
	tip("Germes dérivant avec l'écoulement, lignes tracées dans la direction de la vitesse");
	bool turbid = effects_->turbidity();
	if (ImGui::Checkbox("Eau turbide (crue chargée)", &turbid))
		effects_->setTurbidity(turbid);
	tip("L'eau rapide se charge en sédiments et devient brune, puis décante en eau calme");

	ImGui::SeparatorText("Vue");
	ImGui::SetNextItemWidth(-90);
	ImGui::SliderFloat("exagération", &render_.z_scale, 1.0f, 2.5f * zScale0_, "x %.2f");
	tip("Exagération verticale (les ombres sont recalculées)");
	if (ImGui::Button("Recadrer la caméra"))
		camera_.reset();
	tip("Origine");
	ImGui::TextDisabled("Clic gauche : tourner   molette : zoom\nclic milieu : déplacer   Ctrl+clic : sonde");

	ImGui::SeparatorText("Images");
	if (ImGui::Button("Capture (F12)"))
		captureRequest_ = std::filesystem::path("captures") /
		                  std::format("swe_{:%Y%m%d_%H%M%S}.png", std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now()));
	ImGui::SameLine();
	if (ImGui::Button(recording_ ? "Arrêter ●" : "Enregistrer"))
	{
		recording_ = !recording_;
		if (recording_)
		{
			recordDir_   = std::filesystem::path("captures") /
			             std::format("rec_{:%Y%m%d_%H%M%S}", std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now()));
			recordIndex_ = 0;
			std::filesystem::create_directories(recordDir_);
		}
	}
	tip("Séquence PNG dans captures/rec_*/ (assembler avec : ffmpeg -framerate 30 -i frame_%05d.png out.mp4)");
	ImGui::SetNextItemWidth(-90);
	ImGui::SliderInt("1 image sur", &recordEvery_, 1, 10);
	if (recording_)
		ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.35f, 1.0f), "● %u images -> %s", recordIndex_, recordDir_.string().c_str());
	ImGui::End();
}

void App::drawProbesPanel()
{
	if (!ImGui::Begin(kProbes, &showProbes_))
	{
		ImGui::End();
		return;
	}
	auto const &c = solver_->caseData();
	if (probes_.empty())
		ImGui::TextWrapped("Ctrl + clic sur le terrain pour poser une sonde (8 au plus). Leurs mesures s'affichent ici et "
		                   "dans « Courbes ».");
	int remove = -1;
	ImGuiTableFlags tableFlags = ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_SizingStretchProp;
	if (!probes_.empty() && ImGui::BeginTable("probes", 5, tableFlags))
	{
		ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed, 22.0f);
		ImGui::TableSetupColumn("h (m)");
		ImGui::TableSetupColumn("|u| (m/s)");
		ImGui::TableSetupColumn("eta (m)");
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
	if (!probes_.empty() && ImGui::Button("Tout effacer"))
	{
		probes_.clear();
		updateProbeList();
	}
	ImGui::End();
}

void App::drawPlotsPanel()
{
	if (!ImGui::Begin(kPlots, &showPlots_))
	{
		ImGui::End();
		return;
	}
	auto const &h = history_;
	int         n = static_cast<int>(h.t.size());
	auto const &c = solver_->caseData();
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
		ImGui::EndTabBar();
	}
	ImGui::End();
}

void App::drawPerfPanel()
{
	if (!ImGui::Begin(kPerf, &showPerf_))
	{
		ImGui::End();
		return;
	}
	auto const &io = ImGui::GetIO();
	auto const &c  = solver_->caseData();
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
	ImGui::Checkbox("Démo ImGui", &showDemo_);
	ImGui::End();
}

// Colour bar in the bottom-right corner of the 3D view
void App::drawLegend()
{
	if (render_.colorMode == ColorMode::Realistic)
		return;
	auto const &info  = colorModeInfo(render_.colorMode);
	ImVec2      scale = ImGui::GetIO().DisplayFramebufferScale;
	auto       *vp    = ImGui::GetMainViewport();
	ImVec2      corner(vp->Pos.x + (viewRect_.x + viewRect_.z) / scale.x - 16.0f, vp->Pos.y + (viewRect_.y + viewRect_.w) / scale.y - 16.0f);
	ImGui::SetNextWindowPos(corner, ImGuiCond_Always, ImVec2(1.0f, 1.0f));
	ImGui::SetNextWindowBgAlpha(0.75f);
	ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings |
	                         ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoDocking |
	                         ImGuiWindowFlags_NoMove;
	if (ImGui::Begin("##legend", nullptr, flags))
	{
		if (*info.unit)
			ImGui::Text("%s (%s)", info.label, info.unit);
		else
			ImGui::TextUnformatted(info.label);
		ImVec2      p0 = ImGui::GetCursorScreenPos();
		float       w = 240.0f, hbar = 14.0f;
		auto       *dl = ImGui::GetWindowDrawList();
		int const   steps = 48;
		for (int k = 0; k < steps; ++k)
		{
			glm::vec3 a = turbo(float(k) / steps), b = turbo(float(k + 1) / steps);
			dl->AddRectFilledMultiColor(ImVec2(p0.x + w * k / steps, p0.y), ImVec2(p0.x + w * (k + 1) / steps, p0.y + hbar),
			                            ImGui::GetColorU32(ImVec4(a.r, a.g, a.b, 1)), ImGui::GetColorU32(ImVec4(b.r, b.g, b.b, 1)),
			                            ImGui::GetColorU32(ImVec4(b.r, b.g, b.b, 1)), ImGui::GetColorU32(ImVec4(a.r, a.g, a.b, 1)));
		}
		ImGui::Dummy(ImVec2(w, hbar));
		double offset = render_.colorMode == ColorMode::Elevation ? solver_->caseData().z_ref : 0.0;
		auto   label  = [&](float v) { return std::format("{:.3g}", v + offset); };
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
	auto const &c     = solver_->caseData();
	float       S     = renderer_->S();
	ImVec2      scale = ImGui::GetIO().DisplayFramebufferScale;
	auto       *vp    = ImGui::GetMainViewport();
	glm::mat4   vpm   = camera_.proj(viewRect_.z / std::max(viewRect_.w, 1.0f)) * camera_.view();
	auto       *dl    = ImGui::GetBackgroundDrawList();
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
}

void App::drawHelp()
{
	ImGui::SetNextWindowSize(ImVec2(420, 0), ImGuiCond_FirstUseEver);
	if (ImGui::Begin("Aide", &showHelp_, ImGuiWindowFlags_NoDocking))
	{
		auto row = [](char const *key, char const *what) {
			ImGui::TextColored(ImVec4(0.6f, 0.8f, 1.0f, 1.0f), "%-18s", key);
			ImGui::SameLine(150);
			ImGui::TextUnformatted(what);
		};
		ImGui::SeparatorText("Souris (vue 3D)");
		row("Clic gauche", "tourner autour du point visé");
		row("Clic milieu", "déplacer");
		row("Molette", "zoomer");
		row("Ctrl + clic", "poser une sonde");
		ImGui::SeparatorText("Clavier");
		row("Espace", "pause / reprise");
		row("R", "recommencer");
		row("Origine", "recadrer la caméra");
		row("F12", "capture d'écran (captures/)");
		row("F1", "cette aide");
		ImGui::SeparatorText("Fenêtres");
		ImGui::TextWrapped("Les panneaux se déplacent et s'ancrent par leur barre de titre ; la disposition est "
		                   "mémorisée (imgui.ini). Le menu Fenêtres les rouvre.");
	}
	ImGui::End();
}

// ---------------------------------------------------------------------------------------------
// Probes and picking
// ---------------------------------------------------------------------------------------------

// Ray from the camera through the mouse, marched against the (exaggerated) bed surface.
std::optional<App::Probe> App::pickCell(glm::vec2 mouse) const
{
	auto const &c     = solver_->caseData();
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
	auto const           &c = solver_->caseData();
	std::vector<uint32_t> cells;
	for (auto const &p : probes_)
		cells.push_back(p.i * c.ny + p.j);
	solver_->setProbes(cells);
	for (auto &series : history_.eta)
		std::fill(series.begin(), series.end(), NAN);
	for (auto &series : history_.depth)
		std::fill(series.begin(), series.end(), NAN);
	for (auto &series : history_.speed)
		std::fill(series.begin(), series.end(), NAN);
	for (auto &v : scalars_.probes)
		v = {};
}

// Colour map range following the current maxima (smoothed to avoid flicker)
void App::autoColorRange()
{
	auto const &c = solver_->caseData();
	float       lo = 0.0f, hi = 1.0f;
	switch (render_.colorMode)
	{
	case ColorMode::Depth: hi = std::max(scalars_.hMax, 1e-3f); break;
	case ColorMode::Speed: hi = std::max(scalars_.uMax, 1e-3f); break;
	case ColorMode::Froude: hi = 2.0f; break;
	case ColorMode::Elevation:
	{
		auto [mn, mx] = std::minmax_element(c.zb.begin(), c.zb.end());
		lo            = *mn;
		hi            = std::max(*mn + scalars_.hMax, static_cast<float>(std::min<double>(c.eta_0, *mx)));
		break;
	}
	default: return;
	}
	if (!std::isfinite(lo) || !std::isfinite(hi))
		return;
	bool first      = render_.cmapMax <= render_.cmapMin;
	render_.cmapMin = first ? lo : render_.cmapMin + 0.1f * (lo - render_.cmapMin);
	render_.cmapMax = first ? hi : render_.cmapMax + 0.1f * (hi - render_.cmapMax);
}

}        // namespace swe
