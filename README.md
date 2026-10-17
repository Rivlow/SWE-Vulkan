# SWE-Vulkan

Portage C++20 / Vulkan / Slang du solveur Saint-Venant 2D de `../SWE` (Python/Taichi) :
volumes finis MUSCL + limiteur de Venkatakrishnan, reconstruction hydrostatique d'Audusse, flux HLLC,
Runge-Kutta explicites (euler, ssprk2, ssprk3, rk4, dopri5), frottement de Manning, exutoires.

## Construire

Prérequis : Visual Studio 2026 (MSVC), Vulkan SDK 1.4 (`slangc`), CMake ≥ 3.29, vcpkg
(`C:\Users\lucas\Dev\Fun\Vulkan\Tutorial\vcpkg`, référencé par `CMakePresets.json`).

```bat
build.bat            :: preset debug   -> build\debug\bin\swe.exe
build.bat release    :: preset release -> build\release\bin\swe.exe
```

Les shaders (`shaders/**.slang`) sont compilés au build dans `bin/shaders/*.spv`.

## Lancer

```bat
swe                                   :: fenêtre, cas dam_break
swe --case lake_at_rest --scheme rk4
swe --headless --steps 200 --dump out\run   :: Q.npy, zb.npy, meta.json
swe --headless --selftest
swe --capture out\vue.png --capture-frames 120   :: capture après 120 images puis quitte
swe --sync-validation                 :: validation Vulkan + synchronisation (lent)
swe --help
```

Souris : clic gauche = rotation libre, molette = zoom, clic milieu = translation. Espace : pause, F12 : capture
(`captures\`). Panneau « Contrôles » : pas par image, eau, maillage, exagération verticale, schéma, CFL,
Manning, cas, reset simulation / caméra.

## Rendu

Le terrain, l'eau et le socle sont dessinés directement depuis les buffers du solveur (`Q`, `zb` lus dans les
vertex shaders, aucun maillage CPU), en MSAA 4×. Les couleurs et l'éclairage reprennent les formules de
`ui/renderer.py`, mais par pixel. La transparence de l'eau, que le Python simulait par sommet
(`mix(fond·(0.45+SHALLOW), teinte, opacité)` puis ombrage, Fresnel, reflet, écume), est linéaire en la couleur
du fond : elle est appliquée exactement avec du *dual-source blending* (`image = fond × K + B`), sur le pixel de
terrain réellement rendu derrière.

## Organisation

| Python (`../SWE`)              | C++ / Slang                                        |
|--------------------------------|----------------------------------------------------|
| `solver/reconstruction.py`     | `shaders/solver/reconstruction.slang`              |
| `solver/flux.py`               | `shaders/solver/flux.slang` (+ `flux_x/flux_y`)    |
| `solver/boundary_condition.py` | `shaders/solver/boundary_condition.slang`          |
| `solver/time_integration.py`   | `shaders/solver/solver.slang`, `src/solver/Solver` |
| `solver/runge_kutta.py`        | `src/solver/RungeKutta`                            |
| `main.py` (globales)           | `shaders/solver/params.slang`, `src/solver/SolverParams.h` |
| `cases/*.py`                   | `src/cases/`                                       |

Un pas de temps est un command buffer pré-enregistré (14 dispatches en ssprk2), soumis N fois par image ;
t, dt et s_max remontent au CPU une fois par image, avec une image de retard, sans attente.

## Validation contre le Python

```bat
python tools\validate.py --all        :: 5 schémas sur dam_break vs Taichi
python tools\validate.py dam_break rk4 200
python tools\lake_at_rest.py          :: équilibre du lac au repos (île sèche)
```

`tools/dump_reference.py` exécute le `main.py` d'origine sans le modifier et écrit le même format de dump.

Résultats (M1, RTX 5070 Ti) :

- Euler 50 pas : écart relatif max < 1e-6 sur h, hu, hv ; t et dt identiques.
- ssprk3, rk4 (200 pas) : écart max ~2e-6.
- ssprk2, dopri5 (200 pas) : L1 relatif ~1e-6, mais quelques dizaines de cellules au front de choc / sec-mouillé
  jusqu'à ~1e-3. **Taichi CPU contre Taichi Vulkan montre exactement le même motif** (6 cellules en h, 38 en hu) :
  c'est l'amplification d'arrondi au front, pas un défaut du portage. D'où le critère : L1 ≤ 1e-5 et
  au plus 64 cellules au-dessus de 1e-5.
- Lac au repos (1000 pas) : |eta − 0.5| ≤ 1e-6 ; courants parasites ~1e-5, identiques au Python
  (même cellule, même amplitude : arrondi f32 de l'équilibre hydrostatique).
