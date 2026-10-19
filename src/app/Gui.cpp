// GUI of the application (ImGui + ImPlot): a four-stage workflow (Map, Terrain, Conditions, Simulation) with a
// stepper bar at the top, one side panel whose content follows the stage (sections drawn as cards), the view in
// the remaining space and a plots band under it in the Simulation stage. Fixed layout, computed from the
// viewport every frame. The theme and the fonts are set in App::initImGui (App.cpp), the frame loop in App.cpp.
#include "app/App.h"

#include "util/FileDialog.h"
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

constexpr float kSidePanelWidth = 400.0f;        // x dpi
constexpr float kStageBarHeight = 58.0f;         // x dpi

constexpr ImGuiWindowFlags kFixedWindow = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                                          ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoSavedSettings |
                                          ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;

// palette of the GUI (the widgets' colours come from the style, App.cpp applyTheme)
constexpr ImVec4 kAccent{0.22f, 0.56f, 0.96f, 1.0f};
constexpr ImVec4 kAccentHi{0.32f, 0.64f, 1.0f, 1.0f};
constexpr ImVec4 kAccentLo{0.16f, 0.46f, 0.84f, 1.0f};
constexpr ImVec4 kWarn{1.0f, 0.72f, 0.25f, 1.0f};
constexpr ImVec4 kDanger{0.96f, 0.36f, 0.32f, 1.0f};
constexpr ImVec4 kOk{0.35f, 0.82f, 0.45f, 1.0f};
constexpr ImVec4 kMuted{0.56f, 0.59f, 0.65f, 1.0f};
constexpr ImVec4 kCardBg{0.145f, 0.160f, 0.190f, 1.0f};
constexpr ImVec4 kBarBg{0.085f, 0.095f, 0.11f, 1.0f};

ImVec4 toImVec4(glm::vec3 c, float a = 1.0f)
{
	return ImVec4(c.r, c.g, c.b, a);
}

struct StageInfo
{
	char const *label;
	char const *hint;
};
constexpr StageInfo kStages[] = {
    {"Carte", "Choisir une zone sur la carte et importer son relief, ou ouvrir un cas enregistré"},
    {"Terrain", "Le relief en 3D : exagération verticale, orthophoto, retouches du lit"},
    {"Conditions", "Eau initiale, sol et nappe, pluie, zones et conditions aux limites, sur le maillage"},
    {"Simulation", "Lancer, suivre le bilan, colorer, sonder, tracer des courbes"},
};

constexpr char const *kSideNames[4]  = {"Ouest", "Est", "Sud", "Nord"};
constexpr char const *kSideShort[4]  = {"O", "E", "S", "N"};
constexpr char const *kBcKindNames[] = {"mur", "débit entrant", "niveau imposé", "sortie libre", "hauteur imposée"};
constexpr int         kBcKinds       = int(std::size(kBcKindNames));

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
		return "-";
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

std::string formatLength(double m)
{
	return m >= 1e4 ? std::format("{:.2f} km", m * 1e-3) : std::format("{:.0f} m", m);
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

std::filesystem::path capturePath(char const *prefix)
{
	return paths::dataDir() / "captures" /
	       std::format("{}_{:%Y%m%d_%H%M%S}.png", prefix, std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now()));
}

// ---- widgets ----------------------------------------------------------------------------------------------

// Full-width accent button (the "next step" of each stage)
bool primaryButton(char const *label, float height = 0.0f, float width = -1.0f)
{
	ImGui::PushStyleColor(ImGuiCol_Button, kAccent);
	ImGui::PushStyleColor(ImGuiCol_ButtonHovered, kAccentHi);
	ImGui::PushStyleColor(ImGuiCol_ButtonActive, kAccentLo);
	ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1, 1, 1, 1));
	bool pressed = ImGui::Button(label, ImVec2(width, height));
	ImGui::PopStyleColor(4);
	return pressed;
}

// Small flat button (no background until hovered): toolbars, row actions
bool ghostButton(char const *label, ImVec2 size = ImVec2(0, 0))
{
	ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
	ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(1, 1, 1, 0.08f));
	ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(1, 1, 1, 0.14f));
	bool pressed = ImGui::Button(label, size);
	ImGui::PopStyleColor(3);
	return pressed;
}

// Row of exclusive buttons; returns true when the value changed
bool segmented(char const *id, int &value, std::span<char const *const> labels, float width = -1.0f)
{
	bool  changed = false;
	float avail   = width > 0.0f ? width : ImGui::GetContentRegionAvail().x;
	float spacing = ImGui::GetStyle().ItemInnerSpacing.x;
	float w       = (avail - spacing * float(labels.size() - 1)) / float(labels.size());
	ImGui::PushID(id);
	for (size_t k = 0; k < labels.size(); ++k)
	{
		bool on = int(k) == value;
		if (k > 0)
			ImGui::SameLine(0.0f, spacing);
		ImGui::PushStyleColor(ImGuiCol_Button, on ? kAccent : ImGui::GetStyleColorVec4(ImGuiCol_FrameBg));
		ImGui::PushStyleColor(ImGuiCol_ButtonHovered, on ? kAccentHi : ImGui::GetStyleColorVec4(ImGuiCol_ButtonHovered));
		ImGui::PushStyleColor(ImGuiCol_ButtonActive, on ? kAccentLo : ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
		ImGui::PushStyleColor(ImGuiCol_Text, on ? ImVec4(1, 1, 1, 1) : ImGui::GetStyleColorVec4(ImGuiCol_Text));
		if (ImGui::Button(labels[k], ImVec2(w, 0)) && !on)
		{
			value   = int(k);
			changed = true;
		}
		ImGui::PopStyleColor(4);
	}
	ImGui::PopID();
	return changed;
}

// Small rounded label with a coloured background (status)
void badge(char const *text, ImVec4 color, bool darkText = true)
{
	auto  *dl  = ImGui::GetWindowDrawList();
	ImVec2 p   = ImGui::GetCursorScreenPos();
	ImVec2 ts  = ImGui::CalcTextSize(text);
	float  pad = 7.0f, h = ImGui::GetFrameHeight();
	ImVec2 a(p.x, p.y), b(p.x + ts.x + 2 * pad, p.y + h);
	dl->AddRectFilled(a, b, ImGui::GetColorU32(color), h * 0.5f);
	dl->AddText(ImVec2(a.x + pad, a.y + (h - ts.y) * 0.5f), darkText ? IM_COL32(20, 22, 26, 255) : IM_COL32(245, 246, 248, 255), text);
	ImGui::Dummy(ImVec2(b.x - a.x, h));
}

// Filled dot (colour key)
void dot(ImVec4 color, float r = 5.0f)
{
	auto  *dl = ImGui::GetWindowDrawList();
	ImVec2 p  = ImGui::GetCursorScreenPos();
	float  h  = ImGui::GetFrameHeight();
	dl->AddCircleFilled(ImVec2(p.x + r + 1.0f, p.y + h * 0.5f), r, ImGui::GetColorU32(color));
	ImGui::Dummy(ImVec2(2 * r + 2.0f, h));
}

// Label on the left (muted), value on the right
void kv(char const *label, std::string const &value, ImVec4 valueColor = ImVec4(0, 0, 0, 0))
{
	ImGui::TextDisabled("%s", label);
	float w = ImGui::CalcTextSize(value.c_str()).x;
	ImGui::SameLine(std::max(ImGui::GetCursorPosX() + 8.0f, ImGui::GetContentRegionMax().x - w));
	if (valueColor.w > 0.0f)
		ImGui::TextColored(valueColor, "%s", value.c_str());
	else
		ImGui::TextUnformatted(value.c_str());
}

// Muted small heading inside a card
void sectionLabel(char const *text)
{
	ImGui::Spacing();
	ImGui::TextDisabled("%s", text);
	ImGui::Spacing();
}

// Card: bordered child with a title row (bold) and an accent bar on its left. Collapsible cards remember their
// state in the window's storage. Returns true when the body is to be drawn; always call endCard() afterwards.
struct CardState
{
	bool open;
};

bool beginCard(char const *id, char const *title, ImFont *bold, bool collapsible = false, bool defaultOpen = true,
               char const *rightText = nullptr)
{
	ImGuiID    key   = ImGui::GetID(id);
	auto      *store = ImGui::GetStateStorage();
	bool       open  = collapsible ? store->GetBool(key, defaultOpen) : true;
	ImGui::PushStyleColor(ImGuiCol_ChildBg, kCardBg);
	ImGui::BeginChild(id, ImVec2(0, 0), ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_Borders | ImGuiChildFlags_AlwaysUseWindowPadding);
	ImGui::PopStyleColor();
	auto  *dl   = ImGui::GetWindowDrawList();
	ImVec2 wmin = ImGui::GetWindowPos(), wmax(wmin.x + ImGui::GetWindowWidth(), wmin.y + ImGui::GetWindowHeight());
	dl->AddRectFilled(ImVec2(wmin.x, wmin.y + 10.0f), ImVec2(wmin.x + 3.0f, wmax.y - 10.0f), ImGui::GetColorU32(open ? kAccent : kMuted), 2.0f);
	// title row (clickable when collapsible)
	ImVec2 rowPos  = ImGui::GetCursorScreenPos();
	float  rowH    = ImGui::GetFrameHeight();
	float  rowW    = ImGui::GetContentRegionAvail().x;
	if (collapsible)
	{
		if (ImGui::InvisibleButton("##hdr", ImVec2(rowW, rowH)))
		{
			open = !open;
			store->SetBool(key, open);
		}
		if (ImGui::IsItemHovered())
			dl->AddRectFilled(ImVec2(rowPos.x - 4, rowPos.y), ImVec2(rowPos.x + rowW + 4, rowPos.y + rowH), IM_COL32(255, 255, 255, 10), 4.0f);
		// chevron
		float  cx = rowPos.x + rowW - 10.0f, cy = rowPos.y + rowH * 0.5f;
		ImU32  cc = ImGui::GetColorU32(kMuted);
		if (open)
			dl->AddTriangleFilled(ImVec2(cx - 5, cy - 2.5f), ImVec2(cx + 5, cy - 2.5f), ImVec2(cx, cy + 3.5f), cc);
		else
			dl->AddTriangleFilled(ImVec2(cx - 2.5f, cy - 5), ImVec2(cx - 2.5f, cy + 5), ImVec2(cx + 3.5f, cy), cc);
		ImGui::SetCursorScreenPos(rowPos);
	}
	ImGui::PushFont(bold, 0.0f);
	ImGui::AlignTextToFramePadding();
	ImGui::TextUnformatted(title);
	ImGui::PopFont();
	if (rightText && *rightText)
	{
		float w = ImGui::CalcTextSize(rightText).x;
		ImGui::SameLine(rowW - w - (collapsible ? 22.0f : 0.0f));
		ImGui::AlignTextToFramePadding();
		ImGui::TextDisabled("%s", rightText);
	}
	// (the title text, as high as the invisible button, leaves the cursor under the row: no SetCursorPos here,
	// ImGui wants an item after any cursor move that extends the window)
	if (open)
		ImGui::Spacing();
	return open;
}

void endCard()
{
	ImGui::EndChild();
	ImGui::Spacing();
}

char const *bcUnit(BcKind k)
{
	switch (k)
	{
	case BcKind::Inflow: return "%.1f m³/s";
	case BcKind::Level: return "%.2f m DNG";
	case BcKind::Depth: return "%.2f m";
	default: return "%.2f";
	}
}

bool bcHasValue(BcKind k)
{
	return k == BcKind::Inflow || k == BcKind::Level || k == BcKind::Depth;
}

}        // namespace

// ---------------------------------------------------------------------------------------------
// Frame layout
// ---------------------------------------------------------------------------------------------

void App::drawGui()
{
	highlight_ = HIGHLIGHT_NONE;        // set again by the widgets hovered this frame
	if (autoRange_)                     // the legend stays in presentation mode: keep its range live there too
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
	float  barH = std::round(kStageBarHeight * dpi_);
	drawStageBar(origin.y, barH);

	float  panelW = std::min(std::round(kSidePanelWidth * dpi_), std::round(work.x * 0.42f));
	ImVec2 panelPos(origin.x, origin.y + barH), panelSize(panelW, work.y - barH);
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
			drawViewOverlay();
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
		ImGui::MenuItem("Conditions aux limites dans la vue", nullptr, &render_.showBorders);
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

// Stepper: the four steps as numbered discs joined by a line, the current one filled with the accent, the
// completed ones outlined; status (case, clock, badges) on the right
void App::drawStageBar(float y, float height)
{
	auto *vp = ImGui::GetMainViewport();
	ImGui::SetNextWindowPos(ImVec2(vp->Pos.x, y));
	ImGui::SetNextWindowSize(ImVec2(vp->Size.x, height));
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(16.0f, 0.0f));
	ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
	ImGui::PushStyleColor(ImGuiCol_WindowBg, kBarBg);
	if (ImGui::Begin("##stagebar", nullptr, kFixedWindow | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse))
	{
		auto  *dl     = ImGui::GetWindowDrawList();
		bool   loaded = session_->loaded();
		float  r      = std::round(12.0f * dpi_);
		float  cy     = ImGui::GetWindowPos().y + height * 0.5f;
		float  x      = ImGui::GetWindowPos().x + 16.0f * dpi_ + r;
		float  gap    = std::round(26.0f * dpi_);
		int    cur    = int(stage_);
		ImGui::PushFont(fontBold_, 0.0f);
		for (int k = 0; k < 4; ++k)
		{
			bool   active  = k == cur, done = k < cur, enabled = k == 0 || loaded;
			ImVec4 colV    = !enabled ? ImVec4(0.30f, 0.33f, 0.38f, 1) : active ? kAccent : done ? kAccentHi : kMuted;
			ImU32  col     = ImGui::GetColorU32(colV);
			auto   label   = kStages[k].label;
			ImVec2 ts      = ImGui::CalcTextSize(label);
			float  w       = 2 * r + 8.0f + ts.x;
			// hit area
			ImGui::SetCursorScreenPos(ImVec2(x - r, cy - r - 4.0f));
			ImGui::PushID(k);
			ImGui::BeginDisabled(!enabled);
			if (ImGui::InvisibleButton("##step", ImVec2(w, 2 * r + 8.0f)) && !active)
				setStage(static_cast<Stage>(k));
			ImGui::EndDisabled();
			bool hovered = ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled);
			if (hovered)
				ImGui::SetTooltip("%s", kStages[k].hint);
			ImGui::PopID();
			// disc
			if (active)
				dl->AddCircleFilled(ImVec2(x, cy), r, col, 32);
			else
			{
				if (hovered && enabled)
					dl->AddCircleFilled(ImVec2(x, cy), r, IM_COL32(255, 255, 255, 18), 32);
				dl->AddCircle(ImVec2(x, cy), r - 1.0f, col, 32, 2.0f);
			}
			auto   num = std::to_string(k + 1);
			ImVec2 ns  = ImGui::CalcTextSize(num.c_str());
			dl->AddText(ImVec2(x - ns.x * 0.5f, cy - ns.y * 0.5f), active ? IM_COL32(255, 255, 255, 255) : col, num.c_str());
			// label
			ImU32 tc = !enabled ? ImGui::GetColorU32(ImVec4(0.40f, 0.43f, 0.48f, 1)) : active || hovered ? ImGui::GetColorU32(ImGuiCol_Text) : ImGui::GetColorU32(kMuted);
			dl->AddText(ImVec2(x + r + 8.0f, cy - ts.y * 0.5f), tc, label);
			x += w;
			if (k < 3)
			{
				// connector
				ImU32 lc = ImGui::GetColorU32(k < cur ? kAccentHi : ImVec4(0.26f, 0.29f, 0.34f, 1));
				dl->AddLine(ImVec2(x + 6.0f, cy), ImVec2(x + gap - 6.0f, cy), lc, 2.0f);
				x += gap + r;
			}
		}
		ImGui::PopFont();

		// status, right-aligned: import progress, else case / clock / badges
		float right = ImGui::GetWindowPos().x + ImGui::GetWindowWidth() - 16.0f * dpi_;
		if (map_->importing())
		{
			auto  txt = std::format("Import : {}", map_->progressStage());
			float pw  = std::round(220.0f * dpi_);
			ImGui::SetCursorScreenPos(ImVec2(right - pw, cy - ImGui::GetFrameHeight() * 0.5f));
			ImGui::PushStyleColor(ImGuiCol_PlotHistogram, kAccent);
			ImGui::ProgressBar(map_->progressFraction(), ImVec2(pw, 0), std::format("{:.0f} %", 100.0f * map_->progressFraction()).c_str());
			ImGui::PopStyleColor();
			float tw = ImGui::CalcTextSize(txt.c_str()).x;
			dl->AddText(ImVec2(right - pw - tw - 12.0f, cy - ImGui::GetTextLineHeight() * 0.5f), ImGui::GetColorU32(kMuted), txt.c_str());
		}
		else if (loaded)
		{
			auto const &c    = session_->caseData();
			std::string name = c.label;
			std::string clk  = std::format("t = {}", formatTime(scalars_.t));
			struct Badge { char const *text; ImVec4 color; bool dark; };
			std::vector<Badge> badges;
			if (recording_)
				badges.push_back({"REC", kDanger, false});
			if (stage_ == Stage::Simulation)
				badges.push_back(paused_ ? Badge{"EN PAUSE", kWarn, true} : Badge{"EN COURS", kOk, true});
			else
				badges.push_back({"t = 0", ImVec4(0.26f, 0.29f, 0.34f, 1), false});
			float h = ImGui::GetFrameHeight(), xr = right;
			for (auto it = badges.rbegin(); it != badges.rend(); ++it)
			{
				float bw = ImGui::CalcTextSize(it->text).x + 14.0f;
				xr -= bw;
				ImGui::SetCursorScreenPos(ImVec2(xr, cy - h * 0.5f));
				badge(it->text, it->color, it->dark);
				xr -= 8.0f;
			}
			float cw = ImGui::CalcTextSize(clk.c_str()).x;
			xr -= cw + 8.0f;
			dl->AddText(ImVec2(xr, cy - ImGui::GetTextLineHeight() * 0.5f), ImGui::GetColorU32(kMuted), clk.c_str());
			ImGui::PushFont(fontBold_, 0.0f);
			float nw = ImGui::CalcTextSize(name.c_str()).x;
			float maxW = std::max(xr - x - 60.0f, 80.0f);
			if (nw > maxW)
			{
				while (name.size() > 4 && ImGui::CalcTextSize((name + "...").c_str()).x > maxW)
					name.pop_back();
				name += "...";
				nw = ImGui::CalcTextSize(name.c_str()).x;
			}
			xr -= nw + 16.0f;
			dl->AddText(ImVec2(xr, cy - ImGui::GetTextLineHeight() * 0.5f), ImGui::GetColorU32(ImGuiCol_Text), name.c_str());
			ImGui::PopFont();
		}
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
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12.0f, 12.0f));
	if (ImGui::Begin("##side", nullptr, kFixedWindow))
	{
		switch (stage_)
		{
		case Stage::Map: drawMapPanel(); break;
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
// Stage 1: Map
// ---------------------------------------------------------------------------------------------

void App::drawMapPanel()
{
	if (beginCard("cases", "Cas enregistrés", fontBold_, false, true, std::format("{}", caseList_.size()).c_str()))
	{
		auto const &current = session_->caseData().name;
		float       rows    = std::clamp(float(caseList_.size()), 2.0f, 7.0f);
		ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.11f, 0.12f, 0.145f, 1.0f));
		if (ImGui::BeginListBox("##cases", ImVec2(-1, rows * ImGui::GetTextLineHeightWithSpacing() + 8.0f)))
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
		ImGui::PopStyleColor();
		if (primaryButton(std::format("Ouvrir « {} »", session_->caseData().label).c_str()))
			openCase(current);
		tip("Charge le cas et montre son terrain en 3D");
	}
	endCard();
	map_->drawImportSidebar();
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
	if (beginCard("terrain", "Terrain", fontBold_))
	{
		ImGui::TextWrapped("%s", c.label.c_str());
		ImGui::Spacing();
		kv("Maillage", std::format("{} x {} mailles de {:.3g} m", c.nx, c.ny, c.dx()));
		kv("Mailles", std::format("{:.2f} M", c.cells() * 1e-6));
		kv("Emprise", std::format("{} x {}", formatLength(c.Lx), formatLength(c.Ly)));
		auto [zmin, zmax] = session_->bedRange();
		kv("Relief", std::format("{:.1f} à {:.1f} m DNG", zmin + c.z_ref, zmax + c.z_ref));
		if (transition_.active)
			ImGui::TextColored(kAccentHi, "Extrusion du relief... (clic pour passer)");
	}
	endCard();

	if (beginCard("view", "Vue", fontBold_))
	{
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
		ImGui::SameLine(ImGui::GetContentRegionAvail().x * 0.5f);
		ImGui::Checkbox("Mailles brutes", &render_.blocks);
		tip("Chaque maille dessinée comme une colonne à sommet plat, sans interpolation entre les centres\n"
		    "(18 sommets par maille : plus lourd sur les grandes grilles)");
	}
	endCard();

	if (beginCard("relief", "Relief", fontBold_))
	{
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
	}
	endCard();

	// footer
	float footer = ImGui::GetFrameHeightWithSpacing() * 1.6f;
	ImGui::SetCursorPosY(std::max(ImGui::GetCursorPosY(), ImGui::GetWindowHeight() - footer));
	if (primaryButton("Suivant : Conditions", ImGui::GetFrameHeight() * 1.3f))
		setStage(Stage::Conditions);
}

// ---------------------------------------------------------------------------------------------
// Stage 3: Conditions
// ---------------------------------------------------------------------------------------------

// Selection card: mode, statistics of the selected cells, quick selections
void App::drawSelectionCard()
{
	auto const &c = session_->caseData();
	auto        right = hasSelection() ? std::format("{} maille{}", selStats_.count, selStats_.count > 1 ? "s" : "") : std::string("aucune maille");
	if (!beginCard("selection", "Sélection", fontBold_, false, true, right.c_str()))
	{
		endCard();
		return;
	}
	int mode = int(selectMode_);
	static char const *const kModes[] = {"Rectangle", "Pinceau"};
	float modeW = ImGui::GetContentRegionAvail().x;
	if (selectMode_ == SelectMode::Brush)
		modeW -= 120.0f;
	if (segmented("mode", mode, kModes, modeW))
		selectMode_ = static_cast<SelectMode>(mode);
	tip("Rectangle : glisser dans la vue. Pinceau : peindre les mailles sous le curseur.\n"
	    "Maj + glisser : ajouter, Ctrl + glisser : retirer, clic simple : une maille.");
	if (selectMode_ == SelectMode::Brush)
	{
		ImGui::SameLine();
		ImGui::SetNextItemWidth(-1);
		ImGui::SliderInt("##radius", &brushRadius_, 0, 8, "rayon %d");
		tip("Mailles autour de la maille peinte");
	}

	if (hasSelection())
	{
		auto const &b  = selStats_.bounds;
		uint32_t    ni = b.i1 - b.i0 + 1, nj = b.j1 - b.j0 + 1;
		ImGui::TextDisabled("i %u..%u, j %u..%u  (%u x %u)   %s", b.i0, b.i1, b.j0, b.j1, ni, nj,
		                    formatArea(double(selStats_.count) * c.dx() * c.dy()).c_str());
		ImGui::TextDisabled("fond %.2f à %.2f m DNG", selStats_.zmin + c.z_ref, selStats_.zmax + c.z_ref);
	}
	else
		ImGui::TextDisabled("Glisser dans la vue.  Maj : ajouter   Ctrl : retirer   Échap : effacer");

	// quick actions
	ImGui::Spacing();
	float bw = (ImGui::GetContentRegionAvail().x - 4 * ImGui::GetStyle().ItemInnerSpacing.x) / 5.0f;
	if (ImGui::Button("Tout", ImVec2(bw, 0)))
		selectAll();
	tip("Ctrl + A");
	ImGui::SameLine(0, ImGui::GetStyle().ItemInnerSpacing.x);
	if (ImGui::Button("Inverser", ImVec2(bw, 0)))
		invertSelection();
	tip("Ctrl + I");
	ImGui::SameLine(0, ImGui::GetStyle().ItemInnerSpacing.x);
	ImGui::BeginDisabled(!hasSelection());
	if (ImGui::Button("Effacer", ImVec2(bw, 0)))
		clearSelection();
	tip("Échap");
	ImGui::SameLine(0, ImGui::GetStyle().ItemInnerSpacing.x);
	if (ImGui::Button("+ 1", ImVec2(bw, 0)))
		growSelection(1);
	tip("Étendre d'une maille (dilatation)");
	ImGui::SameLine(0, ImGui::GetStyle().ItemInnerSpacing.x);
	if (ImGui::Button("- 1", ImVec2(bw, 0)))
		growSelection(-1);
	tip("Réduire d'une maille (érosion)");
	ImGui::EndDisabled();

	// by elevation: the river bed, a flood plain under a level...
	if (selectBelow_ == 0.0)
		selectBelow_ = std::round((session_->bedRange().first + c.z_ref) * 10.0) / 10.0 + 1.0;
	ImGui::AlignTextToFramePadding();
	ImGui::TextDisabled("Fond sous");
	ImGui::SameLine();
	ImGui::SetNextItemWidth(140.0f);
	ImGui::InputDouble("##below", &selectBelow_, 0.0, 0.0, "%.2f m DNG");
	ImGui::SameLine();
	if (ImGui::Button("Sélectionner", ImVec2(-1, 0)))
		selectBelow(selectBelow_ - c.z_ref, ImGui::GetIO().KeyShift);
	tip("Mailles dont le fond est sous cette cote (le lit, une plaine inondable...).\nMaj : seulement parmi les mailles déjà sélectionnées.");

	// the boundary cells of a side
	ImGui::AlignTextToFramePadding();
	ImGui::TextDisabled("Bord du domaine");
	ImGui::SameLine();
	float sw = (ImGui::GetContentRegionAvail().x - 3 * ImGui::GetStyle().ItemInnerSpacing.x) / 4.0f;
	for (int s = 0; s < 4; ++s)
	{
		if (s > 0)
			ImGui::SameLine(0, ImGui::GetStyle().ItemInnerSpacing.x);
		ImGui::PushID(s);
		if (ImGui::Button(kSideNames[s], ImVec2(sw, 0)))
			selectSide(s, ImGui::GetIO().KeyShift ? SelectOp::Add : ImGui::GetIO().KeyCtrl ? SelectOp::Remove : SelectOp::Replace);
		if (ImGui::IsItemHovered())
		{
			highlight_ = HIGHLIGHT_SIDE + uint32_t(s);
			ImGui::SetTooltip("Mailles du bord %s (Maj : ajouter, Ctrl : retirer)", kSideNames[s]);
		}
		ImGui::PopID();
	}
	endCard();
}

void App::drawWaterTab()
{
	auto const &c = session_->caseData();
	if (beginCard("water-block", "Bloc d'eau sur la sélection", fontBold_))
	{
		static char const *const kModes[] = {"hauteur (m)", "cote (m DNG)", "ajouter (m)"};
		segmented("wmode", waterMode_, kModes);
		ImGui::SetNextItemWidth(-110);
		ImGui::InputFloat("valeur", &waterValue_, 0.1f, 1.0f, "%.2f");
		ImGui::BeginDisabled(!hasSelection());
		if (primaryButton("Poser l'eau", 0.0f, -90.0f))
		{
			WaterEdit mode  = waterMode_ == 0 ? WaterEdit::Depth : waterMode_ == 1 ? WaterEdit::Level : WaterEdit::Add;
			float     value = waterMode_ == 1 ? waterValue_ - static_cast<float>(c.z_ref) : waterValue_;
			session_->editWater(selection(), mode, value);
			effects_->reset();
		}
		ImGui::SameLine();
		if (ImGui::Button("Vider", ImVec2(-1, 0)))
		{
			session_->editWater(selection(), WaterEdit::Clear, 0.0f);
			effects_->reset();
		}
		ImGui::EndDisabled();
		tip("Eau au repos sur les mailles sélectionnées ; fait partie de l'état initial (Recommencer la restaure)");
	}
	endCard();

	if (beginCard("water-level", "Niveau initial sur tout le domaine", fontBold_))
	{
		if (levelValue_ == 0.0)
			levelValue_ = c.eta_0 + c.z_ref;
		ImGui::SetNextItemWidth(-110);
		ImGui::InputDouble("cote (m DNG)", &levelValue_, 0.5, 5.0, "%.2f");
		if (ImGui::Button("Remplir jusqu'à cette cote", ImVec2(-1, 0)))
		{
			session_->editWater(CellRect{0, 0, c.nx - 1, c.ny - 1}, WaterEdit::Level, static_cast<float>(levelValue_ - c.z_ref), true);
			effects_->reset();
		}
		tip("Lac au repos : h = max(cote - fond, 0) partout");
		ImGui::Checkbox("Eau mince visible", &render_.thinWater);
		tip("Donne une opacité minimale aux lames d'eau de quelques cm, sinon invisibles");
	}
	endCard();
}

void App::drawSoilTab()
{
	auto const &c  = session_->caseData();
	bool        on = c.gw.enabled;
	if (beginCard("aquifer", "Nappe souterraine", fontBold_))
	{
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
	}
	endCard();

	if (beginCard("soil", on ? "Paramètres du sol" : "Paramètres du sol (à l'activation)", fontBold_))
	{
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
			float bw = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) * 0.5f;
			if (ImGui::Button("Appliquer partout", ImVec2(bw, 0)))
			{
				session_->setSoil(CellRect{0, 0, c.nx - 1, c.ny - 1}, soil_);
				onCaseLoaded();
			}
			ImGui::SameLine();
			ImGui::BeginDisabled(!hasSelection());
			if (primaryButton("À la sélection", 0.0f, bw))
			{
				session_->setSoil(selection(), soil_);
				onCaseLoaded();
			}
			ImGui::EndDisabled();
			tip("Sol hétérogène : des valeurs différentes sur les mailles sélectionnées");
		}
	}
	endCard();

	if (beginCard("friction", "Frottement", fontBold_))
	{
		float manning = static_cast<float>(c.manning_n);
		ImGui::SetNextItemWidth(-130);
		if (ImGui::SliderFloat("Manning n", &manning, 0.0f, 0.1f, "%.3f"))
			session_->setManning(manning);
		tip("Frottement de fond (s/m^1/3) : 0.03 lit naturel, 0.04 plaine, 0 sans frottement");
	}
	endCard();
}

// Bridges: a deck over the selected cells (soffit, deck top), the list of the case's decks, the slot width
void App::drawStructuresTab()
{
	auto const &c = session_->caseData();
	if (beginCard("new-bridge", "Pont sur la sélection", fontBold_))
	{
		ImGui::TextWrapped("Tablier au-dessus des mailles sélectionnées. Trois régimes : surface libre sous le tablier, "
		                   "écoulement en charge dès que l'eau touche l'intrados, surverse au-dessus du tablier.");
		if (std::isnan(bridgeRoof_) && hasSelection())
			bridgeRoof_ = std::ceil(selStats_.zmax + c.z_ref + 2.0);        // a first guess: 2 m above the highest bed
		double roof = std::isnan(bridgeRoof_) ? c.z_ref + 2.0 : bridgeRoof_;
		ImGui::SetNextItemWidth(-140);
		if (ImGui::InputDouble("intrados", &roof, 0.1, 1.0, "%.2f m DNG"))
			bridgeRoof_ = roof;
		tip("Cote du dessous du tablier (sous-face, intrados) : l'écoulement passe en charge au-dessus");
		ImGui::SetNextItemWidth(-140);
		ImGui::InputFloat("épaisseur", &bridgeThick_, 0.1f, 0.5f, "%.2f m");
		bridgeThick_ = std::max(bridgeThick_, 0.0f);
		tip("Épaisseur du tablier : l'eau passe par-dessus quand elle dépasse intrados + épaisseur (surverse)");
		ImGui::SetNextItemWidth(-140);
		ImGui::InputTextWithHint("nom", "pont", bridgeLabel_, sizeof(bridgeLabel_));
		if (hasSelection())
		{
			double clear = roof - (selStats_.zmax + c.z_ref), clearMax = roof - (selStats_.zmin + c.z_ref);
			if (clearMax <= 0.0)
				ImGui::TextColored(kWarn, "Intrados sous le fond partout : le tablier est un seuil (surverse seule).");
			else
				ImGui::TextDisabled("hauteur libre %.2f à %.2f m", std::max(clear, 0.0), clearMax);
		}
		ImGui::BeginDisabled(!hasSelection());
		if (primaryButton("Ajouter le pont"))
		{
			session_->addBridge(selection(), roof - c.z_ref, roof - c.z_ref + bridgeThick_, bridgeLabel_[0] ? bridgeLabel_ : "pont");
			bridgeLabel_[0] = 0;
		}
		ImGui::EndDisabled();
		tip("Les mailles prises par un autre pont passent à celui-ci. La simulation repart de t = 0.");
	}
	endCard();

	if (beginCard("bridges", "Ponts du cas", fontBold_, false, true, std::format("{}", c.bridges.size()).c_str()))
	{
		if (c.bridges.empty())
			ImGui::TextDisabled("Aucun pont.");
		else if (ImGui::BeginTable("bridges", 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp))
		{
			ImGui::TableSetupColumn("pont");
			ImGui::TableSetupColumn("intrados", ImGuiTableColumnFlags_WidthFixed, 92);
			ImGui::TableSetupColumn("dessus", ImGuiTableColumnFlags_WidthFixed, 92);
			ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 48);
			ImGui::TableHeadersRow();
			int remove = -1;
			for (size_t b = 0; b < c.bridges.size(); ++b)
			{
				auto const &bd = c.bridges[b];
				ImGui::PushID(int(b));
				ImGui::TableNextRow();
				ImGui::TableNextColumn();
				dot(ImVec4(0.95f, 0.6f, 0.25f, 1));
				ImGui::SameLine();
				ImGui::AlignTextToFramePadding();
				ImGui::TextUnformatted(bd.label.c_str());
				if (ImGui::IsItemHovered())
					ImGui::SetTooltip("%zu mailles", session_->countOf(bd.cells));
				double roof = bd.zRoof + c.z_ref, top = bd.zDeck + c.z_ref;
				ImGui::TableNextColumn();
				ImGui::SetNextItemWidth(-1);
				if (ImGui::InputDouble("##r", &roof, 0.0, 0.0, "%.2f", ImGuiInputTextFlags_EnterReturnsTrue))
					session_->setBridge(b, roof - c.z_ref, std::max(top, roof) - c.z_ref);
				ImGui::TableNextColumn();
				ImGui::SetNextItemWidth(-1);
				if (ImGui::InputDouble("##t", &top, 0.0, 0.0, "%.2f", ImGuiInputTextFlags_EnterReturnsTrue))
					session_->setBridge(b, roof - c.z_ref, std::max(top, roof) - c.z_ref);
				ImGui::TableNextColumn();
				if (ghostButton("sel"))
					setSelection(bd.cells);
				tip("Sélectionner les mailles du pont");
				ImGui::SameLine(0, 2);
				if (ghostButton("x"))
					remove = int(b);
				tip("Supprimer le pont");
				ImGui::PopID();
			}
			ImGui::EndTable();
			if (remove >= 0)
				session_->removeBridge(size_t(remove));
		}
		ImGui::TextDisabled("Cotes en m DNG ; les mailles sous tablier sont teintées en orange (Afficher les zones).");
	}
	endCard();

	if (!c.bridges.empty() && beginCard("bridge-model", "Modèle", fontBold_, true, false))
	{
		float slot = static_cast<float>(c.bridgeSlot);
		ImGui::SetNextItemWidth(-140);
		if (ImGui::SliderFloat("fente", &slot, 0.01f, 1.0f, "%.2f", ImGuiSliderFlags_Logarithmic))
			session_->setBridgeSlot(slot);
		tip("Fente de Preissmann (largeur relative) des mailles en charge : elle garde une célérité finie\n"
		    "(sqrt(g c / fente)) et ne stocke qu'une fine lame au-dessus de l'intrados. 1 = pas de fente : la maille\n"
		    "en charge stocke toute la colonne piézométrique (modèle wolfgpu). Plus petite = plus réaliste, pas de\n"
		    "temps plus court (x 4.5 pour 0.05).");
		ImGui::TextDisabled("Hauteur efficace min(h, c) + max(h - c - e, 0), frottement sur 1, 2 ou 3 parois.");
	}
	endCard();
}

void App::drawZonesTab()
{
	auto const &c = session_->caseData();
	if (beginCard("new-zone", "Nouvelle zone sur la sélection", fontBold_))
	{
		ImGui::TextDisabled("Appliquée à la fin de chaque pas de temps.");
		static char const *const kKinds[] = {"source", "puits", "niveau fixé"};
		segmented("zkind", zoneKind_, kKinds);
		if (zoneKind_ != 1)
		{
			ImGui::SetNextItemWidth(-110);
			ImGui::InputFloat(zoneKind_ == 0 ? "débit" : "cote", &zoneValue_, 1.0f, 10.0f, zoneKind_ == 0 ? "%.2f m³/s" : "%.2f m DNG");
		}
		ImGui::SetNextItemWidth(-110);
		ImGui::InputTextWithHint("nom", "optionnel", zoneLabel_, sizeof(zoneLabel_));
		ImGui::BeginDisabled(!hasSelection() || c.zones.size() >= MAX_ZONES);
		if (primaryButton("Créer la zone"))
		{
			ZoneKind kind  = zoneKind_ == 0 ? ZoneKind::Source : zoneKind_ == 1 ? ZoneKind::Sink : ZoneKind::Level;
			double   value = zoneKind_ == 2 ? zoneValue_ - c.z_ref : zoneValue_;
			std::string label = zoneLabel_[0] ? zoneLabel_ : (zoneKind_ == 0 ? "source" : zoneKind_ == 1 ? "puits" : "niveau");
			session_->addZone(selection(), kind, value, label);
			zoneLabel_[0] = 0;
		}
		ImGui::EndDisabled();
		if (c.zones.size() >= MAX_ZONES)
			ImGui::TextColored(kWarn, "%u zones au plus", MAX_ZONES);
	}
	endCard();

	if (beginCard("zones", "Zones du cas", fontBold_, false, true, std::format("{}", c.zones.size()).c_str()))
	{
		if (c.zones.empty())
			ImGui::TextDisabled("Aucune zone.");
		else if (ImGui::BeginTable("zones", 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp))
		{
			ImGui::TableSetupColumn("zone");
			ImGui::TableSetupColumn("type", ImGuiTableColumnFlags_WidthFixed, 56);
			ImGui::TableSetupColumn("valeur", ImGuiTableColumnFlags_WidthFixed, 116);
			ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 24);
			ImGui::TableHeadersRow();
			int remove = -1;
			for (size_t z = 0; z < c.zones.size(); ++z)
			{
				auto const &zd = c.zones[z];
				ImGui::PushID(int(z));
				ImGui::TableNextRow();
				ImGui::TableNextColumn();
				ImVec4 col = zd.kind == ZoneKind::Source ? ImVec4(0.35f, 0.6f, 1, 1) : zd.kind == ZoneKind::Sink ? ImVec4(1, 0.4f, 0.3f, 1) : ImVec4(0.3f, 0.95f, 0.9f, 1);
				dot(col);
				ImGui::SameLine();
				ImGui::AlignTextToFramePadding();
				ImGui::TextUnformatted(zd.label.c_str());
				if (ImGui::IsItemHovered())
					ImGui::SetTooltip("mailles (%u..%u, %u..%u)", zd.i0, zd.i1, zd.j0, zd.j1);
				ImGui::TableNextColumn();
				ImGui::AlignTextToFramePadding();
				ImGui::TextUnformatted(zd.kind == ZoneKind::Source ? "source" : zd.kind == ZoneKind::Sink ? "puits" : "niveau");
				ImGui::TableNextColumn();
				if (zd.kind != ZoneKind::Sink)
				{
					double v = zd.kind == ZoneKind::Level ? zd.value + c.z_ref : zd.value;
					ImGui::SetNextItemWidth(-1);
					if (ImGui::InputDouble("##v", &v, 0.0, 0.0, zd.kind == ZoneKind::Level ? "%.2f m" : "%.1f m³/s", ImGuiInputTextFlags_EnterReturnsTrue))
						session_->setZoneValue(z, zd.kind == ZoneKind::Level ? v - c.z_ref : v);
				}
				ImGui::TableNextColumn();
				if (ghostButton("x"))
					remove = int(z);
				tip("Supprimer la zone");
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
	endCard();
}

// Schematic of the domain with its four sides coloured by condition, and one row per side
void App::drawDomainSides()
{
	auto const &c = session_->caseData();
	// schematic
	float  availW = ImGui::GetContentRegionAvail().x;
	float  boxH   = std::round(110.0f * dpi_);
	ImVec2 origin = ImGui::GetCursorScreenPos();
	auto  *dl     = ImGui::GetWindowDrawList();
	float  margin = 26.0f;
	float  ratio  = float(c.Lx / std::max(c.Ly, 1e-9));
	float  w      = availW - 2 * margin, h = boxH - 2 * margin;
	if (w / h > ratio)
		w = h * ratio;
	else
		h = w / ratio;
	ImVec2 a(origin.x + (availW - w) * 0.5f, origin.y + (boxH - h) * 0.5f), b(a.x + w, a.y + h);
	dl->AddRectFilled(a, b, IM_COL32(255, 255, 255, 10), 3.0f);
	// sides: west (left), east (right), south (bottom), north (top)
	ImVec2 const p0[4] = {ImVec2(a.x, a.y), ImVec2(b.x, a.y), ImVec2(a.x, b.y), ImVec2(a.x, a.y)};
	ImVec2 const p1[4] = {ImVec2(a.x, b.y), ImVec2(b.x, b.y), ImVec2(b.x, b.y), ImVec2(b.x, a.y)};
	ImVec2 const lab[4] = {ImVec2(a.x - 16.0f, (a.y + b.y) * 0.5f), ImVec2(b.x + 8.0f, (a.y + b.y) * 0.5f),
	                       ImVec2((a.x + b.x) * 0.5f, b.y + 4.0f), ImVec2((a.x + b.x) * 0.5f, a.y - 18.0f)};
	int hovered = -1;
	for (int s = 0; s < 4; ++s)
	{
		// hit zone around the edge
		ImVec2 hmin(std::min(p0[s].x, p1[s].x) - 8.0f, std::min(p0[s].y, p1[s].y) - 8.0f);
		ImVec2 hmax(std::max(p0[s].x, p1[s].x) + 8.0f, std::max(p0[s].y, p1[s].y) + 8.0f);
		ImGui::SetCursorScreenPos(hmin);
		ImGui::PushID(s);
		if (ImGui::InvisibleButton("##side", ImVec2(hmax.x - hmin.x, hmax.y - hmin.y)))
			bcSide_ = s;
		if (ImGui::IsItemHovered())
			hovered = s;
		ImGui::PopID();
		auto  &bc    = c.bc[s];
		ImVec4 col   = toImVec4(bcColor(uint32_t(bc.kind)));
		bool   sel   = s == bcSide_, hov = s == hovered;
		float  thick = sel ? 7.0f : 5.0f;
		if (sel || hov)
			dl->AddLine(p0[s], p1[s], IM_COL32(255, 255, 255, hov ? 160 : 90), thick + 4.0f);
		dl->AddLine(p0[s], p1[s], ImGui::GetColorU32(col), thick);
		// label: short name + value
		std::string txt = kSideShort[s];
		if (bc.kind == BcKind::Inflow)
			txt += std::format(" {:.1f} m³/s", bc.value);
		else if (bc.kind == BcKind::Level)
			txt += std::format(" {:.2f} m", bc.value + c.z_ref);
		else if (bc.kind == BcKind::Depth)
			txt += std::format(" h {:.2f} m", bc.value);
		else if (bc.kind == BcKind::Free)
			txt += " libre";
		ImVec2 ts = ImGui::CalcTextSize(txt.c_str());
		ImVec2 tp = lab[s];
		if (s == 0)
			tp.x -= ts.x;
		if (s < 2)
			tp.y -= ts.y * 0.5f;
		else
			tp.x -= ts.x * 0.5f;
		dl->AddText(tp, ImGui::GetColorU32(sel ? ImGuiCol_Text : ImGuiCol_TextDisabled), txt.c_str());
	}
	if (hovered >= 0)
	{
		highlight_ = HIGHLIGHT_SIDE + uint32_t(hovered);
		ImGui::SetTooltip("%s : %s", kSideNames[hovered], kBcKindNames[int(c.bc[hovered].kind)]);
	}
	else
		highlight_ = HIGHLIGHT_SIDE + uint32_t(bcSide_);
	ImGui::SetCursorScreenPos(ImVec2(origin.x, origin.y + boxH));
	ImGui::Spacing();

	// the selected side: kind, value, facts
	auto bc = c.bc[bcSide_];
	ImGui::PushFont(fontBold_, 0.0f);
	ImGui::AlignTextToFramePadding();
	ImGui::Text("Bord %s", kSideNames[bcSide_]);
	ImGui::PopFont();
	ImGui::SameLine();
	int  kind    = int(bc.kind);
	bool changed = false;
	ImGui::SetNextItemWidth(-1);
	if (ImGui::Combo("##kind", &kind, kBcKindNames, kBcKinds) && kind != int(bc.kind))
	{
		bc.kind  = static_cast<BcKind>(kind);
		bc.value = bc.kind == BcKind::Level ? c.eta_0 : bc.kind == BcKind::Inflow ? 10.0 : bc.kind == BcKind::Depth ? 1.0 : 0.0;
		changed  = true;
	}
	if (bcHasValue(bc.kind))
	{
		double v = bc.kind == BcKind::Level ? bc.value + c.z_ref : bc.value;
		ImGui::SetNextItemWidth(-1);
		if (ImGui::InputDouble("##value", &v, 0.0, 0.0, bcUnit(bc.kind), ImGuiInputTextFlags_EnterReturnsTrue))
		{
			bc.value = bc.kind == BcKind::Level ? v - c.z_ref : v;
			changed  = true;
		}
		tip(bc.kind == BcKind::Inflow ? "Débit total réparti sur le bord au prorata de h^(5/3) (Entrée pour valider)"
		    : bc.kind == BcKind::Level ? "Cote de surface libre imposée (Entrée pour valider)"
		                               : "Hauteur d'eau imposée sur le bord (Entrée pour valider)");
	}
	if (changed)
		session_->setBoundary(bcSide_, bc);
	// facts about the side: lowest point (where an inflow enters), length
	float    zmin = 1e30f;
	uint32_t n    = bcSide_ < 2 ? c.ny : c.nx;
	for (uint32_t k = 0; k < n; ++k)
		zmin = std::min(zmin, c.zb[bcSide_ < 2 ? c.index(bcSide_ == 0 ? 0 : c.nx - 1, k) : c.index(k, bcSide_ == 2 ? 0 : c.ny - 1)]);
	ImGui::TextDisabled("%u mailles, %s, point bas %.2f m DNG", n, formatLength(bcSide_ < 2 ? c.Ly : c.Lx).c_str(), zmin + c.z_ref);
	if (bc.kind == BcKind::Inflow)
		ImGui::TextDisabled("Bord sec : l'eau entre par les mailles sous %.2f m DNG", zmin + c.z_ref + 0.5);
	if (bc.kind == BcKind::Level && bc.value < zmin)
		ImGui::TextColored(kWarn, "Niveau sous le point bas du bord : il se comporte comme une sortie.");
}

// Weak conditions on the faces of the selected cells
void App::drawFaceConditions()
{
	auto const &c = session_->caseData();
	ImGui::TextWrapped("La face choisie des mailles sélectionnées devient un bord du domaine : son flux est calculé "
	                   "pour imposer la condition (mur, débit total, niveau, hauteur ou sortie libre). De l'autre côté "
	                   "d'une face intérieure, la maille voisine voit un mur.");
	ImGui::Spacing();
	ImGui::AlignTextToFramePadding();
	ImGui::TextDisabled("Face");
	ImGui::SameLine(60.0f);
	segmented("fside", faceSide_, kSideNames);
	tip("Seules les faces tournées vers l'extérieur de la sélection sont prises (son enveloppe de ce côté)");
	ImGui::AlignTextToFramePadding();
	ImGui::TextDisabled("Type");
	ImGui::SameLine(60.0f);
	ImGui::SetNextItemWidth(-1);
	if (ImGui::Combo("##fkind", &faceKind_, kBcKindNames, kBcKinds))
		faceValue_ = faceKind_ == int(BcKind::Level) ? c.eta_0 + c.z_ref : faceKind_ == int(BcKind::Inflow) ? 10.0 : 1.0;
	auto fk = static_cast<BcKind>(faceKind_);
	if (bcHasValue(fk))
	{
		ImGui::AlignTextToFramePadding();
		ImGui::TextDisabled("Valeur");
		ImGui::SameLine(60.0f);
		ImGui::SetNextItemWidth(-1);
		ImGui::InputDouble("##fvalue", &faceValue_, fk == BcKind::Inflow ? 1.0 : 0.1, fk == BcKind::Inflow ? 10.0 : 1.0, bcUnit(fk));
		tip(fk == BcKind::Inflow ? "Débit total du groupe de faces, réparti au prorata de h^(5/3) (comme un bord)"
		    : fk == BcKind::Level ? "Cote de surface libre imposée faiblement (flux vers la cote)"
		                          : "Hauteur d'eau imposée faiblement (flux vers la hauteur)");
	}
	ImGui::AlignTextToFramePadding();
	ImGui::TextDisabled("Nom");
	ImGui::SameLine(60.0f);
	ImGui::SetNextItemWidth(-1);
	ImGui::InputTextWithHint("##flabel", "optionnel", faceLabel_, sizeof(faceLabel_));
	ImGui::BeginDisabled(!hasSelection() || c.faceBcs.size() >= MAX_FACE_BC);
	if (primaryButton(std::format("Imposer sur le bord {} de la sélection", kSideNames[faceSide_]).c_str()))
	{
		double      value = fk == BcKind::Level ? faceValue_ - c.z_ref : faceValue_;
		std::string label = faceLabel_[0] ? faceLabel_ : std::format("{} {}", kBcKindNames[faceKind_], kSideShort[faceSide_]);
		if (session_->addFaceBc(selection(), static_cast<Side>(faceSide_), fk, value, label))
			renderer_->setFaceBcs(session_->caseData());
		faceLabel_[0] = 0;
	}
	ImGui::EndDisabled();
	if (!hasSelection())
		ImGui::TextDisabled("Sélectionner des mailles d'abord.");
	else if (c.faceBcs.size() >= MAX_FACE_BC)
		ImGui::TextColored(kWarn, "%u groupes au plus", MAX_FACE_BC);

	if (!c.faceBcs.empty())
	{
		sectionLabel("Conditions sur faces du cas");
		if (ImGui::BeginTable("facebcs", 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp))
		{
			ImGui::TableSetupColumn("groupe");
			ImGui::TableSetupColumn("faces", ImGuiTableColumnFlags_WidthFixed, 50);
			ImGui::TableSetupColumn("valeur", ImGuiTableColumnFlags_WidthFixed, 116);
			ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 50);
			ImGui::TableHeadersRow();
			int remove = -1;
			for (size_t g = 0; g < c.faceBcs.size(); ++g)
			{
				auto const &f = c.faceBcs[g];
				ImGui::PushID(int(g));
				ImGui::TableNextRow();
				ImGui::TableNextColumn();
				dot(toImVec4(bcColor(uint32_t(f.kind))));
				ImGui::SameLine();
				ImGui::AlignTextToFramePadding();
				ImGui::TextUnformatted(f.label.c_str());
				if (ImGui::IsItemHovered())
				{
					highlight_ = HIGHLIGHT_GROUP + uint32_t(g);
					ImGui::SetTooltip("%s", kBcKindNames[int(f.kind)]);
				}
				ImGui::TableNextColumn();
				ImGui::AlignTextToFramePadding();
				ImGui::Text("%zu", f.faces.size());
				ImGui::TableNextColumn();
				if (bcHasValue(f.kind))
				{
					double v = f.kind == BcKind::Level ? f.value + c.z_ref : f.value;
					ImGui::SetNextItemWidth(-1);
					if (ImGui::InputDouble("##v", &v, 0.0, 0.0, bcUnit(f.kind), ImGuiInputTextFlags_EnterReturnsTrue))
						session_->setFaceBc(g, f.kind, f.kind == BcKind::Level ? v - c.z_ref : v);
				}
				else
				{
					ImGui::AlignTextToFramePadding();
					ImGui::TextDisabled("%s", kBcKindNames[int(f.kind)]);
				}
				ImGui::TableNextColumn();
				if (ghostButton("sel"))
					setSelection(session_->faceBcCells(g));
				tip("Sélectionner les mailles du groupe");
				ImGui::SameLine(0, 2);
				if (ghostButton("x"))
					remove = int(g);
				tip("Supprimer le groupe");
				ImGui::PopID();
			}
			ImGui::EndTable();
			if (remove >= 0)
			{
				session_->removeFaceBc(size_t(remove));
				renderer_->setFaceBcs(session_->caseData());
			}
		}
	}
}

void App::drawBoundariesTab()
{
	if (beginCard("sides", "Bords du domaine", fontBold_))
		drawDomainSides();
	endCard();
	if (beginCard("facebc", "Conditions sur les faces de la sélection", fontBold_))
		drawFaceConditions();
	endCard();
	if (beginCard("bclegend", "Légende", fontBold_, true, false))
	{
		for (int k = 0; k < kBcKinds; ++k)
		{
			dot(toImVec4(bcColor(uint32_t(k))));
			ImGui::SameLine();
			ImGui::AlignTextToFramePadding();
			ImGui::TextUnformatted(kBcKindNames[k]);
			if (k + 1 < kBcKinds && (k % 2 == 0))
				ImGui::SameLine(ImGui::GetContentRegionAvail().x * 0.5f + 20.0f);
		}
		ImGui::TextDisabled("Les murs du domaine ne sont dessinés que survolés.");
	}
	endCard();
}

void App::drawConditionsPanel()
{
	drawSelectionCard();
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
			if (ImGui::BeginTabItem("Ouvrages"))
			{
				drawStructuresTab();
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
		tip("Écrit le lit, l'eau initiale, le sol, les zones et les conditions dans assets/");
	}
	else
		ImGui::TextDisabled("Cas analytique : les conditions ne sont pas enregistrées.");
	if (primaryButton("Lancer la simulation", ImGui::GetFrameHeight() * 1.3f))
		setStage(Stage::Simulation);
}

// Rain / infiltration zones and their table of discharges over time (wolfgpu's add_rain / add_infiltration)
void App::drawChronologyEditor()
{
	using Kind = Session::ChronologyKind;
	auto const &c = session_->caseData();
	if (!beginCard("chrono", "Chronologies", fontBold_))
	{
		endCard();
		return;
	}
	static char const *const kKinds[] = {"infiltration", "pluie"};
	segmented("ckind", chronoKind_, kKinds, ImGui::GetContentRegionAvail().x - 110.0f);
	tip("Débits par zone au cours du temps, comme add_rain / add_infiltration de wolfgpu.\n"
	    "Positif : apport d'eau en surface ; une infiltration négative retire au plus l'eau présente.");
	ImGui::SameLine();
	ImGui::Checkbox("en mm/h", &chronoMmh_);
	tip("Coché : lame d'eau par heure ; décoché : m³/s pour toute la zone (convention wolfgpu)");
	Kind        kind  = chronoKind_ == 1 ? Kind::Rain : Kind::Infiltration;
	auto const &ch    = kind == Kind::Rain ? c.rain : c.infiltration;
	auto        areas = session_->chronologyZoneAreas(kind);
	char const *unit  = chronoMmh_ ? "%.4g mm/h" : "%.4g m³/s";
	auto        toUser   = [&](double q, double area) { return chronoMmh_ ? (area > 0.0 ? q / area * 3.6e6 : 0.0) : q; };
	auto        fromUser = [&](double v, double area) { return chronoMmh_ ? v * area / 3.6e6 : v; };

	// new zone
	ImGui::SetNextItemWidth(-110);
	ImGui::InputFloat("valeur", &chronoValue_, 1.0f, 10.0f, unit);
	bool added = false;
	auto add   = [&](CellMask const &m) {
        double area = double(session_->countOf(m)) * c.dx() * c.dy();
        session_->addChronologyZone(kind, m, fromUser(chronoValue_, area));
        added = true;
	};
	float bw = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) * 0.5f;
	ImGui::BeginDisabled(!hasSelection());
	if (primaryButton("Zone = sélection", 0.0f, bw))
		add(selection());
	ImGui::EndDisabled();
	ImGui::SameLine();
	if (ImGui::Button("Zone = domaine", ImVec2(bw, 0)))
		add(CellMask(c.cells(), 1));
	tip("La valeur s'applique à toutes les lignes de la table ; une maille appartient à une seule zone de chaque type");
	if (added)
	{
		endCard();
		return;        // the zone count and areas computed above are stale: the table is drawn next frame
	}

	if (!ch.active())
	{
		ImGui::TextDisabled("aucune zone de %s", kind == Kind::Rain ? "pluie" : "infiltration");
		endCard();
		return;
	}
	// table: one row per line (start time), one column per zone
	size_t const nz = ch.zoneCount();
	auto         t = ch.t;
	auto         q = ch.q;
	bool         lerp = ch.lerp, changed = false;
	int          removeZone = -1, removeLine = -1;
	int          cols = int(std::min<size_t>(nz, 62)) + 2;
	sectionLabel("Table des débits");
	if (ImGui::BeginTable("chrono", cols, ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollX | ImGuiTableFlags_SizingFixedFit,
	                      ImVec2(0, std::min(30.0f + 26.0f * float(t.size() + 1), 240.0f))))
	{
		ImGui::TableSetupScrollFreeze(1, 1);
		ImGui::TableSetupColumn("t (s)", 0, 70);
		for (int z = 0; z + 2 < cols; ++z)
			ImGui::TableSetupColumn(std::format("zone {}", z + 1).c_str(), 0, 84);
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
				ImGui::SetTooltip("%.0f m²", areas[z]);
			ImGui::SameLine();
			if (ghostButton("x"))
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
			if (t.size() > 1 && ghostButton("x"))
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
	ImGui::TextDisabled("valeurs en %s, Entrée pour valider", chronoMmh_ ? "mm/h" : "m³/s par zone");
	endCard();
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
		if (primaryButton("Lancer", ImGui::GetFrameHeight() * 1.3f))
			paused_ = false;
	}
	else
	{
		ImGui::PushStyleColor(ImGuiCol_Button, kWarn);
		ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(1.0f, 0.80f, 0.40f, 1.0f));
		ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.9f, 0.62f, 0.18f, 1.0f));
		ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.08f, 0.09f, 0.1f, 1.0f));
		if (ImGui::Button("Pause", ImVec2(-1, ImGui::GetFrameHeight() * 1.3f)))
			paused_ = true;
		ImGui::PopStyleColor(4);
	}
	tip("Espace");
	if (ImGui::Button("Recommencer", ImVec2(bw, 0)))
		resetSimulation();
	tip("R : remet l'état initial (eau, nappe, zones conservés)");
	ImGui::SameLine();
	if (ImGui::Button("Conditions", ImVec2(bw, 0)))
		setStage(Stage::Conditions);
	tip("Retour à l'étape Conditions (la simulation repart de t = 0)");
	ImGui::SetNextItemWidth(-90);
	ImGui::SliderInt("pas / image", &stepsPerFrame_, 1, 500, "%d", ImGuiSliderFlags_AlwaysClamp | ImGuiSliderFlags_Logarithmic);
	tip("Nombre de pas de temps calculés entre deux images : plus = simulation plus rapide, image moins fluide");
}

// Recording of h, qx, qy to a .swerec file (sim/Recorder)
void App::drawRecordSection()
{
	bool active = recorder_ && recorder_->active();
	if (!active)
	{
		ImGui::SetNextItemWidth(-130);
		ImGui::InputDouble("tous les (s simulées)", &recordInterval_, 0.1, 1.0, "%.2f");
		recordInterval_ = std::max(recordInterval_, 0.0);
		tip("Intervalle de temps simulé entre deux enregistrements (0 = à chaque image)");
		ImGui::Checkbox("16 bits (moitié de place)", &recordHalf_);
		tip("Décoché : h, qx, qy en float32, identiques au solveur (12 octets par maille).\n"
		    "Coché : h au mm (uint16), qx, qy en float16 (6 octets par maille) : pour les longues séquences.");
		// folder: text field (applied on Enter / focus loss), native picker, explorer
		ImGui::SetNextItemWidth(-92);
		if (ImGui::InputTextWithHint("##recdir", "dossier des enregistrements", swerecDirBuf_, sizeof(swerecDirBuf_),
		                             ImGuiInputTextFlags_EnterReturnsTrue) ||
		    (ImGui::IsItemDeactivatedAfterEdit()))
			setRecordDir(std::filesystem::path(swerecDirBuf_));
		tip("Dossier où les fichiers .swerec sont écrits (gardé dans swe_settings.json ; --record-dir en ligne de\n"
		    "commande). Vide = records/ à côté de l'exécutable.");
		ImGui::SameLine();
		if (ImGui::Button("…"))
			if (auto dir = dialog::pickFolder(swerecDir_, L"Dossier des enregistrements"))
				setRecordDir(*dir);
		tip("Choisir le dossier");
		ImGui::SameLine();
		if (ImGui::Button("Ouvrir"))
			if (!dialog::openInExplorer(swerecDir_))
				log::warn("cannot open {}", swerecDir_.string());
		tip("Ouvrir le dossier dans l'explorateur");
		auto path = defaultRecordPath();
		ImGui::TextDisabled("%s", path.filename().string().c_str());
		tip(path.string().c_str());
		if (primaryButton("Démarrer l'enregistrement"))
			startRecording(path, recordInterval_, recordHalf_);
		tip("h, qx, qy des tuiles mouillées, copiés par le GPU et écrits par un thread de fond ;\n"
		    "lecture : tools/swerec.py, post-traitement : tools/swepost.py. L'enregistrement s'arrête en quittant\n"
		    "l'étape Simulation.");
		return;
	}
	auto st = recorder_->stats();
	kv("Fichier", recorder_->path().filename().string());
	kv("Dossier", recorder_->path().parent_path().string());
	kv("Enregistrements", std::format("{}   (dernier t = {})", st.records, formatTime(st.lastT)));
	kv("Taille", std::format("{:.1f} MB", double(st.bytes) / 1048576.0));
	kv("Intervalle", std::format("{:.3g} s   ({})", recorder_->every(), recorder_->halfPrecision() ? "16 bits" : "float32"));
	if (st.pending > 0 || st.delayed > 0)
		kv("File d'attente", std::format("{} en cours, {} différé(s)", st.pending, st.delayed), st.delayed > 0 ? kWarn : ImVec4(0, 0, 0, 0));
	tip("Différé : aucun tampon libre à l'image voulue (le disque ne suit pas), l'enregistrement est pris à l'image suivante");
	ImGui::PushStyleColor(ImGuiCol_Button, kDanger);
	ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(1.0f, 0.45f, 0.4f, 1.0f));
	ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.85f, 0.3f, 0.26f, 1.0f));
	if (ImGui::Button("Arrêter l'enregistrement", ImVec2(-1, 0)))
		stopRecording();
	ImGui::PopStyleColor(3);
}

void App::drawBalanceSection()
{
	auto const &c = session_->caseData();
	kv("Temps", std::format("{}   ({} pas)", formatTime(scalars_.t), scalars_.n_steps));
	kv("Pas de temps", std::format("{:.3e} s", scalars_.dt));
	kv("Célérité max", std::format("{:.3f} m/s", scalars_.s_max));
	kv("Cadence", std::format("{:.0f} pas/s", stepsPerSec_));
	if (!std::isfinite(scalars_.t) || !std::isfinite(scalars_.dt) || !std::isfinite(scalars_.volume))
		ImGui::TextColored(kDanger, "La simulation a divergé : Recommencer");

	sectionLabel("Surface");
	double dV = volume0_ > 0.0 ? (scalars_.volume - volume0_) / volume0_ * 100.0 : 0.0;
	kv("Volume", std::format("{}   ({:+.4f} %)", formatVolume(scalars_.volume), dV));
	tip("Variation de volume depuis t = 0 : 0 avec des murs partout (conservation), sinon débits entrants / sortants");
	kv("Emprise mouillée", formatArea(scalars_.wetCells * c.dx() * c.dy()));
	kv("h max", std::format("{:.3f} m", scalars_.hMax));
	kv("|u| max", std::format("{:.3f} m/s", scalars_.uMax));
	if (c.drain_passes > 0)
		kv("Limiteur", std::format("{} maille(s)", scalars_.limitedCells));
	else
		kv("Limiteur", "désactivé (h < 0 tronqué)", kMuted);
	tip("Mailles dont les flux sortants ont été réduits pour garder h >= 0 (dernier étage du dernier pas)");
	if (scalars_.clampCells > 0)
	{
		kv("h < 0", std::format("{} troncature(s), max {:.2e} m", scalars_.clampCells, scalars_.clampMax), kWarn);
		tip("Hauteurs négatives ramenées à 0 depuis t = 0 (masse créée). Limiteur actif : seulement des résidus d'arrondi");
	}

	if (c.gw.enabled)
	{
		sectionLabel("Nappe");
		kv("Volume", formatVolume(scalars_.gwVolume));
		tip("Eau stockée dans l'aquifère : somme de Sy h_gw dx dy");
		kv("Infiltré", formatVolume(scalars_.gwInfil));
		kv("Exfiltré", formatVolume(scalars_.gwExfil));
		kv("Échange", std::format("{:+.4g} m³/s", scalars_.gwRate));
		tip("Débit net surface -> nappe au dernier pas (> 0 : infiltration)");
	}
	bool chronologies = c.infiltration.active() || c.rain.active();
	if (chronologies)
	{
		sectionLabel("Pluie / infiltration");
		kv("Pluie", formatVolume(scalars_.srcRain));
		kv("Apport net", formatVolume(scalars_.srcNet));
		tip("Volume ajouté à la surface par les chronologies (pluie + infiltration signée), depuis t = 0.\n"
		    "Une infiltration négative ne retire au plus que l'eau présente.");
	}
	if (c.gw.enabled || chronologies)
	{
		double total0 = balance0_;
		double total  = double(scalars_.volume) + scalars_.gwVolume - scalars_.srcNet;
		kv("Bilan total", std::format("{:+.4f} % depuis t = 0", total0 > 0.0 ? (total - total0) / total0 * 100.0 : 0.0));
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
		ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed, 24.0f);
		ImGui::TableSetupColumn("h (m)");
		ImGui::TableSetupColumn("|u| (m/s)");
		ImGui::TableSetupColumn("eta (m)");
		if (c.gw.enabled)
			ImGui::TableSetupColumn("nappe (m)");
		ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 24.0f);
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
			if (ghostButton("x"))
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

	sectionLabel("Éléments");
	float colW = ImGui::GetContentRegionAvail().x / 3.0f;
	ImGui::Checkbox("Eau", &render_.showWater);
	ImGui::SameLine(colW);
	ImGui::Checkbox("Maillage", &render_.showMesh);
	ImGui::SameLine(2 * colW);
	ImGui::Checkbox("Zones", &render_.showZones);
	ImGui::Checkbox("Lignes de courant", &render_.showLines);
	tip("Germes dérivant avec l'écoulement, lignes tracées dans la direction de la vitesse");
	ImGui::SameLine(2 * colW);
	ImGui::Checkbox("Eau mince", &render_.thinWater);
	tip("Opacité minimale des lames d'eau de quelques cm");
	bool turbid = effects_->turbidity();
	if (ImGui::Checkbox("Eau turbide (crue chargée)", &turbid))
		effects_->setTurbidity(turbid);
	tip("L'eau rapide se charge en sédiments et devient brune, puis décante en eau calme");
	ImGui::Checkbox("Conditions aux limites", &render_.showBorders);
	tip("Bords ouverts du domaine et conditions sur faces, colorés par type");
	ImGui::SameLine(2 * colW);
	ImGui::Checkbox("Mailles brutes", &render_.blocks);
	tip("Colonnes à sommet plat, sans interpolation : la valeur de chaque maille telle que le solveur la voit");
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
		sectionLabel("Souterrain");
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
	sectionLabel("Coupe");
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
		ImGui::SameLine();
		if (hasSelection() && ImGui::SmallButton("Sélection"))
			render_.crop = {selStats_.bounds.i0, selStats_.bounds.j0, selStats_.bounds.i1, selStats_.bounds.j1};
	}

	sectionLabel("Vue");
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
	if (beginCard("run", c.label.c_str(), fontBold_, false, true, std::format("{} x {} · {:.3g} m", c.nx, c.ny, c.dx()).c_str()))
		drawTransportSection();
	endCard();
	if (beginCard("balance", "Temps et bilan", fontBold_, true, true))
		drawBalanceSection();
	endCard();
	{
		bool rec = recorder_ && recorder_->active();
		if (beginCard("record", "Enregistrement", fontBold_, true, false, rec ? "REC" : nullptr))
			drawRecordSection();
		endCard();
	}
	if (beginCard("probes", "Sondes", fontBold_, true, !probes_.empty(), probes_.empty() ? nullptr : std::format("{}", probes_.size()).c_str()))
		drawProbesSection();
	endCard();
	if (beginCard("display", "Affichage", fontBold_, true, true))
		drawDisplaySection();
	endCard();
	if (beginCard("numerics", "Schéma numérique", fontBold_, true, false))
		drawAdvancedSection();
	endCard();
}

// ---------------------------------------------------------------------------------------------
// Plots band (Simulation stage)
// ---------------------------------------------------------------------------------------------

void App::drawPlotsBand(ImVec2 pos, ImVec2 size)
{
	ImGui::SetNextWindowPos(pos);
	ImGui::SetNextWindowSize(size);
	ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10.0f, 6.0f));
	if (!ImGui::Begin("##plots", nullptr, kFixedWindow))
	{
		ImGui::End();
		ImGui::PopStyleVar(2);
		return;
	}
	// top edge line
	auto *dl = ImGui::GetWindowDrawList();
	dl->AddLine(pos, ImVec2(pos.x + size.x, pos.y), ImGui::GetColorU32(ImGuiCol_Border), 1.0f);
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
	ImGui::PopStyleVar(2);
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
		line("terrain", zb, ImVec4(0.85f, 0.80f, 0.72f, 1.0f), 2.0f);
		if (c.gw.enabled)
			line("cote de la nappe", head, ImVec4(0.45f, 0.70f, 1.0f, 1.0f), 2.0f);
		ImPlot::EndPlot();
	}
}

// ---------------------------------------------------------------------------------------------
// Floating windows and overlays
// ---------------------------------------------------------------------------------------------

void App::drawPerfWindow()
{
	ImGui::SetNextWindowSize(ImVec2(380, 0), ImGuiCond_FirstUseEver);
	if (!ImGui::Begin("Performances", &showPerf_))
	{
		ImGui::End();
		return;
	}
	auto const &io = ImGui::GetIO();
	auto const &c  = session_->caseData();
	ImGui::TextUnformatted(ctx_->properties.deviceName.data());
	kv("Images", std::format("{:.1f} /s   ({:.2f} ms)", io.Framerate, 1000.0f / io.Framerate));
	kv("Pas", std::format("{:.0f} /s   {:.0f} M mailles.pas/s", stepsPerSec_, stepsPerSec_ * c.cells() * 1e-6));
	if (timestampPeriod_ > 0.0)
	{
		sectionLabel("GPU par image");
		double perStep = stepsPerFrame_ > 0 && !paused_ ? gpu_.solver / stepsPerFrame_ * 1000.0 : 0.0;
		kv("solveur", std::format("{:7.3f} ms  ({:.1f} us/pas)", gpu_.solver, perStep));
		kv("effets", std::format("{:7.3f} ms", gpu_.effects));
		kv("rendu", std::format("{:7.3f} ms", gpu_.render));
		tip("Rendu 3D + interface, MSAA 4x");
	}
	else
		ImGui::TextDisabled("Timestamps GPU non disponibles");
	sectionLabel("Rendu");
	kv("vue", std::format("{:.0f} x {:.0f} px, MSAA {}x", viewRect_.z, viewRect_.w, static_cast<unsigned>(renderer_->samples())));
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
	ImGui::SetNextWindowBgAlpha(0.82f);
	ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings |
	                         ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoMove;
	ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.0f);
	if (ImGui::Begin("##legend", nullptr, flags))
	{
		if (*info.unit)
			ImGui::Text("%s (%s)", info.label, info.unit);
		else
			ImGui::TextUnformatted(info.label);
		ImVec2    p0 = ImGui::GetCursorScreenPos();
		float     w = 240.0f, hbar = 12.0f;
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
		ImGui::TextDisabled("%s", lo.c_str());
		ImGui::SameLine(w * 0.5f - ImGui::CalcTextSize(mid.c_str()).x * 0.5f + ImGui::GetStyle().WindowPadding.x);
		ImGui::TextDisabled("%s", mid.c_str());
		ImGui::SameLine(w - ImGui::CalcTextSize(hi.c_str()).x + ImGui::GetStyle().WindowPadding.x);
		ImGui::TextDisabled("%s", hi.c_str());
	}
	ImGui::End();
	ImGui::PopStyleVar();
}

// Readout over the 3D view (top-left): the cell under the mouse in the Conditions stage
void App::drawViewOverlay()
{
	if (stage_ != Stage::Conditions || !hoverCell_)
		return;
	auto const &c = session_->caseData();
	ImVec2      scale = ImGui::GetIO().DisplayFramebufferScale;
	auto       *vp    = ImGui::GetMainViewport();
	ImVec2      corner(vp->Pos.x + viewRect_.x / scale.x + 14.0f, vp->Pos.y + viewRect_.y / scale.y + 14.0f);
	ImGui::SetNextWindowPos(corner, ImGuiCond_Always);
	ImGui::SetNextWindowBgAlpha(0.72f);
	ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings |
	                         ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoInputs;
	ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.0f);
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10.0f, 6.0f));
	if (ImGui::Begin("##hover", nullptr, flags))
	{
		size_t k = c.index(hoverCell_->i, hoverCell_->j);
		ImGui::Text("maille (%u, %u)", hoverCell_->i, hoverCell_->j);
		ImGui::SameLine();
		ImGui::TextDisabled("  fond %.2f m DNG   h0 %.2f m%s", c.zb[k] + c.z_ref, c.h0[k],
		                    selMask_.size() == c.cells() && selMask_[k] ? "   sélectionnée" : "");
	}
	ImGui::End();
	ImGui::PopStyleVar(2);
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
		dl->AddCircleFilled(ImVec2(pos.x, pos.y - 30.0f), 10.0f, IM_COL32(20, 22, 26, 200));
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
	ImGui::SetNextWindowSize(ImVec2(520, 0), ImGuiCond_FirstUseEver);
	if (ImGui::Begin("Aide", &showHelp_))
	{
		auto row = [](char const *key, char const *what) {
			ImGui::TextColored(kAccentHi, "%s", key);
			ImGui::SameLine(190);
			ImGui::TextUnformatted(what);
		};
		ImGui::SeparatorText("Étapes");
		ImGui::TextWrapped("1 Carte : tracer une zone et importer le relief, ou ouvrir un cas. 2 Terrain : vérifier le relief. "
		                   "3 Conditions : eau, sol, pluie, zones et bords sur le maillage, à t = 0. 4 Simulation : lancer et "
		                   "observer. La barre du haut permet de revenir à une étape précédente.");
		ImGui::SeparatorText("Souris (vue 3D)");
		row("Clic gauche", "tourner");
		row("Clic milieu", "déplacer");
		row("Molette", "zoomer");
		row("Ctrl + clic", "poser une sonde (étape Simulation)");
		ImGui::SeparatorText("Sélection de mailles (étape Conditions)");
		row("Glisser", "rectangle de mailles, ou coup de pinceau (mode Pinceau)");
		row("Clic", "une maille");
		row("Maj + glisser / clic", "ajouter à la sélection");
		row("Ctrl + glisser / clic", "retirer de la sélection");
		row("Ctrl + A / Ctrl + I", "tout sélectionner / inverser");
		row("Échap", "effacer la sélection");
		ImGui::SeparatorText("Carte");
		row("Glisser", "déplacer la carte");
		row("Clic droit + glisser", "tracer la zone à importer");
		ImGui::SeparatorText("Clavier");
		row("Espace", "pause / reprise (étape Simulation)");
		row("R", "recommencer (étape Simulation)");
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
