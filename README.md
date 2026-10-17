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

## Interface

Panneaux ancrables (disposition par défaut au premier lancement, puis mémorisée dans `imgui.ini`) ; la vue 3D
occupe la zone centrale libre.

| Panneau            | Contenu                                                                                   |
|--------------------|-------------------------------------------------------------------------------------------|
| Simulation         | cas, pause / recommencer, pas par image, temps simulé, facteur temps réel, bilan de volume, emprise, maxima |
| Schéma numérique   | Runge-Kutta, CFL, Manning, conditions aux limites par bord (mur / débit / niveau), à chaud  |
| Affichage          | coloration (réaliste, h, \|u\|, Froude, cote) + légende, plage auto, eau / maillage / lignes / turbidité, exagération, captures, enregistrement |
| Sondes             | Ctrl+clic sur le terrain (8 max) : h, \|u\|, η, marqueurs numérotés en 3D                    |
| Édition            | vue du dessus, sélection d'un rectangle de mailles en glissant dans la vue 3D, bloc d'eau (hauteur / cote / ajout) posé dans l'état courant et l'état initial, **zones** par maille (source de débit, puits, niveau fixé) appliquées à chaque pas, eau mince visible |
| Courbes            | ImPlot : niveau, hauteur, vitesse des sondes, volume et emprise, pas de temps               |
| Performances       | timestamps GPU par image (solveur / effets / rendu), pas/s, Mmailles·pas/s                  |

Souris : clic gauche = rotation libre, molette = zoom, clic milieu = translation, Ctrl+clic = sonde.
Clavier : Espace pause, R recommencer, Origine recadrer, F12 capture (`captures\`), F1 aide.
Fichier → Enregistrer : séquence PNG (`captures\rec_*\`, écrite en tâche de fond), à assembler avec
`ffmpeg -framerate 30 -i frame_%05d.png out.mp4`. Les réglages (dernier cas, pas par image, coloration…) sont
gardés dans `swe_settings.json`.

Options utiles pour scripter des vues : `--color 0..4`, `--probe i,j` (répétable), `--mesh`, `--z-scale f`,
`--steps-per-frame n`, `--capture fichier.png --capture-frames n`, `--top`, `--water i0,j0,i1,j1,h`,
`--zone source|sink|level,valeur,i0,j0,i1,j1`.

**Conditions aux limites sur un domaine sec** : le débit entrant d'un bord est réparti sur les mailles de bord au
prorata de h^(5/3) ; quand le bord est sec, la répartition utilise la profondeur fictive obtenue en remplissant
le bord jusqu'à 0,5 m au-dessus de son point le plus bas (l'eau entre par le lit de la rivière). Le flux de masse
imposé est **exactement** le débit (`legacy_inflow: true` dans un cas JSON rétablit le flux HLLC de l'état fantôme
du Python, gardé pour la validation du cas `meuse`). Vérification : `inflow_test` (5 m³/s sur un domaine sec)
donne 5,001 m³/s (`tools\volume_check.py`).

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

## Import géographique (panneau « Carte »)

Choisir une zone sur une carte (tuiles OpenStreetMap ou orthophoto SPW), la sonder, l'importer : le relief le
plus fin disponible est assemblé par priorité, la bathymétrie SPW des rivières y est fusionnée, une orthophoto
SPW est drapée sur le terrain, et le cas est sauvé (`assets/cases/<nom>.json` + `assets/data/<nom>.npz` +
`_ortho.png`) pour un rechargement sans réseau.

| Source | Accès | Résolution | Vertical |
|---|---|---|---|
| **MNT / bathymétrie SPW locaux** (dossiers de GeoTIFF ou zips SPW indexés dans « Données locales ») | fichiers | 0,5 – 1 m | DNG |
| Flandre DHMV II | WCS geo.api.vlaanderen.be | 1 m | TAW ≈ DNG |
| Pays-Bas AHN4 | WCS PDOK (sortie en Lambert 72) | 0,5 m | NAP → DNG (+2,32 m) |
| Allemagne NRW DGM1 | WCS geobasis.nrw (sortie en Lambert 72) | 1 m | DHHN2016 → DNG (+2,31 m) |
| France RGE ALTI | WMS-R IGN (GeoTIFF float) | 1 m | NGF-IGN69 → DNG (+1,82 m) |
| Monde : Copernicus GLO-30 | COG public S3 (`/vsicurl/`) | 30 m (MNS : bâtiments inclus) | EGM2008 → DNG (≈ +2,3 m) |

La Wallonie n'a **pas de flux d'altitude** : ses services WMS ne servent que des images 8 bits. Le MNT LiDAR
wallon (1 m 2013-2014, 0,5 m 2021-2022) se télécharge par province sur le portail SPW (zips GeoTIFF, 13 à 43 Go
pour Liège) et se dépose dans un dossier indexé par « Données locales » ; en attendant, Copernicus 30 m sert de
repli. La bathymétrie SPW (voies navigables + lacs, 50 cm) est la **seule** source du fond des rivières : le
LiDAR mesure la surface de l'eau.

Pas de Google Maps : les conditions de Google Maps Platform interdisent de stocker ou d'afficher ses tuiles /
altitudes hors de leur SDK. Les tuiles OSM sont utilisées selon leur politique (User-Agent, pas de téléchargement
massif, attribution affichée) ; données SPW sous CC BY 4.0 (« Source : SPW »), DHMV « Bron: Digitaal Vlaanderen »,
AHN CC0, NRW dl-de/zero, IGN Etalab 2.0, Copernicus avec mention DLR/Airbus/ESA. Les attributions sont écrites
dans le JSON du cas.

**Lit de rivière sans bathymétrie** (panneau Carte → « Lit de rivière ») : tracer l'axe de la rivière sur la
carte (clics, double-clic pour finir), choisir largeur, profondeur et forme des berges, « Creuser le lit » : le
fond est abaissé sous la surface d'eau mesurée par le LiDAR avec un profil parabolique ; « zone à abaisser »
applique un décalage constant dans un polygone. Le cas modifié se garde avec « Enregistrer le cas ».

En ligne de commande :

```bat
swe --import 234800,145600,236400,147200 --cell 2 --library "C:\...\BATHYMETRIE_GEOTIFF_31370_PROV_LIEGE" --name liege --ortho --level 60
swe --import 238500,141500,240500,143500 --cell 2 --name vesdre_chaudfontaine --ortho --carve "240400,141600;239200,142800" --carve-width 40 --carve-depth 4
swe --case liege
```

Validation : `tools\extract_ref.py` (fenêtre rasterio moyennée par blocs) contre `swe --import --dump` :
écart max 0,9 mm sur la bathymétrie SPW rééchantillonnée à 2 m.

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
