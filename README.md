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

## Cas

Chaque cas est un fichier `assets/cases/<nom>.json` (paramètres de `main.py`), chargeable à chaud depuis le
panneau « Cas » ; les données sont les caches de `../SWE/data` copiés dans `assets/data`.

| Cas            | Type           | Données                               | Particularités                          |
|----------------|----------------|---------------------------------------|-----------------------------------------|
| `dam_break`    | `dam_break`    | analytique 250×250                    | murs                                    |
| `lake_at_rest` | `lake_at_rest` | analytique 250×250                    | test d'équilibre, île sèche             |
| `meuse`        | `river`        | `bathy_235577_146334_600_2.npy` 300×300 | débit entrant 600 m³/s, niveau aval, Manning |
| `vesdre`       | `dtm_lake`     | `dtm_vesdre_all.npz` ×2 → 874×548     | lac initial, exutoire, hors-bassin grisé |

Le prétraitement numpy (`fill_outside`, `refine_bilinear`, `refine_nearest`, `z_ref`, exutoire) est porté en
C++ (`src/cases/GridOps`) et redonne exactement le même `zb`. Les conditions aux limites (mur / débit entrant /
niveau imposé), le CFL et le Manning se modifient à chaud.

Note : `vesdre.json` reprend `lake_level = 1000` de `main.py`, ce qui noie tout le bassin (le commentaire du
Python évoque plutôt ~10 m d'eau sur un lac à 290 m, soit `lake_level ≈ 300`).

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
- Meuse (100 pas, débit + niveau + Manning) : `zb` identique, écart max ~3e-6.
- Vesdre (100 pas, 0,48 M mailles) : `zb` identique, L1 ~1e-6, ~100 cellules de front au-dessus de 1e-5,
  comme Taichi CPU contre Taichi Vulkan (94). Tolérance de front : max(64, 0,05 % des mailles).
