# SWE-Vulkan

Portage C++20 / Vulkan / Slang du solveur Saint-Venant 2D de `../SWE` (Python/Taichi) :
volumes finis MUSCL + limiteur de Venkatakrishnan, reconstruction hydrostatique d'Audusse, flux HLLC,
Runge-Kutta SSP (euler, ssprk2, ssprk3) en forme de Shu-Osher, limiteur de vidange (h ≥ 0 sans créer de masse),
frottement de Manning, exutoires.

## Construire

Prérequis : Visual Studio 2026 (MSVC), Vulkan SDK 1.4 (`slangc`), CMake ≥ 3.29, vcpkg
(`C:\Users\lucas\Dev\Fun\Vulkan\Tutorial\vcpkg`, référencé par `CMakePresets.json`).

```bat
build.bat            :: preset debug   -> build\debug\bin\swe.exe
build.bat release    :: preset release -> build\release\bin\swe.exe
```

Les shaders (`shaders/**.slang`) sont compilés au build dans `bin/shaders/*.spv`.

`build.bat` passe la console en UTF-8 (`chcp 65001`). Sans cela, Ninja ne suit aucun en-tête avec un Visual Studio
français : le préfixe des lignes `/showIncludes` (« Remarque : inclusion du fichier : », avec des espaces
insécables) est écrit par CMake en cp850 alors que `cl` l'émet en UTF-8. Modifier un `.h` laissait alors des
objets périmés, d'où des plantages. Pour un build lancé hors de `build.bat`, faire `chcp 65001` avant
`cmake --preset`. `ninja -t deps <objet>` doit lister des en-têtes.

## Distribuer (installateur Windows)

```bat
winget install JRSoftware.InnoSetup      :: une fois
packaging\make_installer.bat             :: build release + dossier build\release\dist + build\release\SWE-Vulkan-<version>-setup.exe
```

L'installateur (≈ 27 Mo) s'installe **par utilisateur** dans `%LOCALAPPDATA%\Programs\SWE Vulkan` (pas de droits
administrateur), avec un raccourci dans le menu Démarrer. L'application y est autonome : `SWE-Vulkan.exe`, les
DLL (GDAL, PROJ, curl…), `shaders/`, `assets/` (cas analytiques et `zone_importee` avec son orthophoto), `proj/`
et `gdal/`. Elle écrit ses réglages, ses captures et les cas importés à côté de l'exécutable. Chez le destinataire :
Windows 10/11 64 bits et des pilotes graphiques récents (Vulkan). Sans fenêtre de console au double-clic ; lancée
depuis un terminal ou par les outils Python, elle y écrit ses journaux comme avant.

- Nom, icône et version : `packaging/app.rc` (ressource Windows, icône `packaging/icon.ico`) et `assets/icon.png`
  (icône de fenêtre) ; le contenu de `dist/` est la cible CMake `dist` (`CMakeLists.txt`), l'installateur est
  `packaging/installer.iss`.
- Pour un zip portable à la place : compresser `build\release\dist`.
- **Partage** : Teams refuse les `.exe` et Discord limite à 25 Mo. Envoyer un **lien** (OneDrive, Google Drive,
  WeTransfer) ; ou `packaging\make_installer.bat light` produit `SWE-Vulkan-<version>-light-setup.exe`, sans le cas
  importé (cas analytiques seulement, les zones s'importent en ligne), qui tient sous 25 Mo.

## Lancer

```bat
swe                                   :: fenêtre, étape Carte (dernier cas chargé en pause)
swe --case zone_importee              :: ouvre le cas : étape Terrain, extrusion animée du relief
swe --case rain_infiltration --run    :: simule tout de suite (captures, scripts)
swe --case lake_at_rest --scheme ssprk3
swe --headless --steps 200 --dump out\run   :: Q.npy, zb.npy, meta.json (+ bilan de volume, h min, limiteur)
swe --case drain_test --drain 0             :: limiteur de vidange coupé (h < 0 tronqué, comme le Python)
swe --headless --selftest
swe --capture out\vue.png --capture-frames 120   :: capture après 120 images puis quitte
swe --sync-validation                 :: validation Vulkan + synchronisation (lent)
swe --help
```

## Interface

L'interface suit le travail en **quatre étapes**, dans la barre du haut. Un seul panneau à gauche change de
contenu avec l'étape ; la vue (carte 2D, puis 3D) occupe le reste. Tout démarre **en pause** : rien ne tourne
avant « Lancer la simulation ».

| Étape | Vue | Panneau |
|---|---|---|
| **1 Carte** | carte 2D (OSM / orthophoto) plein cadre | cas enregistrés (double-clic ou « Ouvrir ») ; nouvelle zone : rectangle tracé sur la carte, maille, nom, orthophoto, « Importer le terrain » ; « Avancé » : sources de données, lit de rivière, rasters locaux |
| **2 Terrain** | relief 3D | infos du cas, vue (dessus / 3D animées, exagération, orthophoto, maillage), retour à la carte pour creuser le lit, « Suivant » |
| **3 Conditions** | 3D avec le **maillage**, outil de sélection actif (glisser) | sélection courante ; onglets **Eau** (bloc d'eau, niveau initial), **Sol** (nappe : kv, kh, Sy, épaisseur, profondeur de nappe, partout ou sur la sélection ; Manning), **Pluie** (chronologies pluie / infiltration), **Zones** (source, puits, niveau), **Bords** (mur / débit / niveau) ; « Enregistrer le cas », « Lancer la simulation » |
| **4 Simulation** | 3D, bande **Courbes** en bas | Lancer / Pause, Recommencer, retour aux Conditions, pas par image ; Temps et bilan (volume, nappe, pluie, bilan total) ; Sondes ; Affichage (coloration + légende, éléments, souterrain, coupe, vue) ; Schéma numérique (replié) |

Après un import (ou l'ouverture d'un cas avec orthophoto), la carte s'estompe sur la vue du dessus du relief,
puis la caméra bascule en 3D pendant que le relief s'extrude ; un clic passe l'animation. Revenir à une étape
précédente se fait en cliquant dans la barre ; revenir aux Conditions remet t = 0.

Souris (vue 3D) : clic gauche = rotation (étape Conditions : sélection de mailles), molette = zoom, clic milieu =
translation, Ctrl+clic = sonde. Carte : glisser = déplacer, clic droit + glisser = zone à importer.
Clavier : Espace pause et R recommencer (étape Simulation), Échap annule la sélection, Origine recadre la
caméra, F12 capture (`captures\`), F11 masque l'interface, F1 aide.
Fichier → Enregistrer une séquence : PNG dans `captures\rec_*\` (écrits en tâche de fond), à assembler avec
`ffmpeg -framerate 30 -i frame_%05d.png out.mp4`. Les réglages (dernier cas, vue de la carte, pas par image,
coloration, bande Courbes…) sont gardés dans `swe_settings.json` ; la disposition est fixe (plus d'`imgui.ini`).

Options utiles pour scripter des vues : `--stage carte|terrain|conditions|simulation` (étape de départ : carte,
ou terrain avec `--case`), `--run` (démarre à l'étape Simulation sans pause), `--color 0..10`, `--probe i,j`
(répétable), `--mesh`, `--z-scale f`, `--steps-per-frame n`, `--capture fichier.png --capture-frames n`, `--top`,
`--water i0,j0,i1,j1,h`, `--zone source|sink|level,valeur,i0,j0,i1,j1`, `--rain` / `--infiltration
i0,j0,i1,j1,mm_h`, `--gw-xray`, `--gw-lines`, `--gw-contours m`, `--crop i0,j0,i1,j1`, `--show-profile`.

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
| `gw_dam_break`, `gw_seepage`, `gw_wolfgpu`, `rain_infiltration` | `dam_break` | analytique 250×250 | nappe, pluie et infiltration (voir « Nappe et infiltration ») |

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

**Projection de travail** : Lambert 72 autour de la Belgique, sinon la zone UTM du centre de la zone
(EPSG:326xx / 327xx), choisie automatiquement ; les cas importés gardent leur projection dans le JSON (`crs`).
Hors Belgique, le relief vient des flux nationaux disponibles (Flandre, Pays-Bas, NRW, France) puis de
Copernicus 30 m partout ; l'orthophoto vient du service régional (SPW, Flandre, PDOK, NRW, IGN) ou, partout dans
le monde, de **Sentinel-2 cloudless** (EOX, 10 m). La carte propose aussi un fond satellite Sentinel-2 mondial.
Exemple : `swe --import-ll 12.455,41.885,12.505,41.915 --cell 10 --name rome --ortho` (Rome, UTM 33N).

**Vitesse de téléchargement.** Les flux WCS sont interrogés à la résolution utile (la moitié de la maille demandée,
jamais plus fin que le produit : à 5 m on demande 2,5 m, soit 6 fois moins d'octets qu'à 1 m ; WCS 2.0.1 via
`SCALEFACTOR`, avec repli automatique sur des tuiles natives si le serveur le refuse), par tuiles de 1024 px au plus,
six requêtes en parallèle ; les mosaïques d'orthophotos sont téléchargées de la même façon. Le sondage (« Sonder »)
ne lit plus que les rasters locaux : les services en ligne n'y figurent que par leur emprise. Toutes les réponses
HTTP sont mises en cache dans `cache/http` (une zone déjà importée se recharge sans réseau). Ordres de grandeur :
5 × 4,5 km à 5 m en Flandre ≈ 3 s, en NRW ≈ 30 s (serveur plus lent), une dalle Copernicus 1 à 2 s.

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

Un pas de temps est un command buffer pré-enregistré (13 phases séparées par des barrières en ssprk2 avec 4 passes
de limiteur par étage), soumis N fois par image ; t, dt et s_max remontent au CPU une fois par image, avec une
image de retard, sans attente. Les noyaux de flux écrivent aussi le terme source de pente (dans `F_R.w` /
`G_U.w`), les flux de bord (un groupe par côté, réduction du débit entrant comprise) tournent dans la même phase
que les flux intérieurs, le dernier étage réduit le s_max du pas suivant en écrivant l'état, le lit `zb` voyage
dans `Q.w` (une lecture par maille), et t est sommé exactement en virgule fixe 64 bits (une somme f32 dérivait de
3,4 s sur 6 h simulées).

### Tuiles actives

Les noyaux de flux, du limiteur et d'Euler ne tournent que sur les **tuiles actives** (16 × 16 mailles, un
groupe par tuile, dispatch indirect) : une tuile est active si elle ou l'une de ses 8 voisines contenait de l'eau
au début du pas, ou si elle porte une zone source / niveau, ou longe un bord à débit ou niveau imposé. Un étage
d'Euler ne déplace l'eau que d'une maille (une face entre deux mailles sèches a un flux exactement nul), donc au
plus 3 mailles par pas : une tuile inactive reste exactement sèche et les faces qui la bordent valent exactement
zéro. Le résultat est **identique au bit près** au calcul de toute la grille (`--all-tiles`, vérifié sur 9
scénarios : fronts qui traversent les tuiles, sources, niveaux, puits, débit entrant sur domaine sec, les 3
schémas). L'humidité des tuiles est tamponnée par le dernier étage avec le numéro du pas (pas de remise à zéro),
la liste est compactée par `build_tiles` (un atomique par wave, le dernier groupe publie le compte).

| Cas (release, RTX 5070 Ti, ssprk2, 4 passes)  | toute la grille | tuiles actives |
|-----------------------------------------------|-----------------|----------------|
| zone_importee 2,3 M mailles, sec              | 2 433 µs/pas    | 23 µs/pas      |
| zone_importee + bloc d'eau 60 × 60 (4 % actif)| 2 432 µs/pas    | 59 µs/pas      |
| vesdre 0,48 M mailles (lac, ~70 % actif)       | 212 µs/pas      | 107 µs/pas     |
| dam_break 62 500 mailles (tout mouillé)       | 46 µs/pas       | 46 µs/pas      |

`swe --headless --case <cas> --steps n --profile 300` donne le temps GPU de chaque phase d'un pas.

Modules (une bibliothèque statique par dossier, chacune liée seulement aux couches du dessous) :

| Dossier       | Responsabilité                                                                         |
|---------------|-----------------------------------------------------------------------------------------|
| `src/util`    | journal, écriture PNG                                                                   |
| `src/vk`      | contexte Vulkan, buffers, images, pipelines                                             |
| `src/cases`   | description d'un cas (`Case.h`, `CaseTypes.h`), lecture / écriture des cas d'`assets/`  |
| `src/geo`     | projections, sources raster / WCS, tuiles, construction du lit, creusement (`BedEdit`)  |
| `src/solver`  | solveur GPU : `SolverParams` (cas → paramètres GPU, fonction pure), `Solver` (buffers, pas enregistré) |
| `src/render`  | effets visuels et rendu 3D, en lecture du set du solveur                                |
| `src/sim`     | `Session` : le cas édité et son état GPU (paramètres, eau, zones, lit), `saveImport`    |
| `src/app`     | fenêtre, boucle d'image, GUI, options de ligne de commande, mode batch                  |

## Positivité : limiteur de vidange

Sans précaution, une maille presque vide peut se voir retirer plus d'eau qu'elle n'en contient en un pas
(reconstruction MUSCL au front, CFL élevé, films minces sur pente raide) : h < 0, et le `max(h, 0)` qui suit
**crée de la masse** (cas `drain_test`, rupture de barrage à CFL 0,9 : +6,5e-4 en volume relatif en 1000 pas ;
lac réel de la Vesdre à CFL 0,9 : +140 %). Le limiteur de vidange remplace cette troncature :

- chaque maille reçoit un facteur α ∈ [0, 1] qui multiplie **tous ses flux de masse sortants** (les deux côtés
  d'une face voient le même flux : conservation exacte) ;
- α résout le point fixe monotone α_i = min(1, (h_i + dt·I_i(α)) / (dt·O_i)), O_i = sortie brute, I_i = entrée
  acceptée par les mailles amont, par K passes de Jacobi (`drain_passes`, 4 par défaut, `--drain n`, slider
  « Drainage ») démarrées à α = 0 : chaque passe est déjà positive, les passes ne font qu'admettre plus de flux.
  K = 1 est le « draining time step » local (seul le volume propre peut sortir) ; K > 1 laisse aussi passer
  l'eau reçue au même pas. C'est la formule de correction α de wolfgpu, itérée vers le haut au lieu de vers le bas ;
- les schémas en temps sont en forme de Shu-Osher, Q ← a·Qⁿ + b·E(Q) avec a, b ≥ 0, calculée comme
  Qⁿ + b·(E(Q) − Qⁿ) : la positivité d'un pas d'Euler se transmet exactement, même en flottant, et sans biais de
  masse quand a + b ≠ 1 en f32 (1/3 + 2/3 de ssprk3 : écrit a·h + b·h, le volume dérivait de 3e-8 par pas).
  rk4 et Dormand-Prince, qui n'ont pas cette forme, ont été retirés ;
- une maille limitée reçoit exactement h = dt·(I_final − I_passe), le zéro mathématique étant écrit tel quel : l'erreur
  de fermeture est de l'ordre d'un ulp du volume de la maille. Les troncatures résiduelles (une maille non limitée
  pile à la marge) sont comptées (`clamp_cells`, `clamp_max`, panneau Bilan) : zéro sur tous les cas testés.

Coût : un dispatch léger par passe et par étage. La passe 0 rassemble les 4 flux de masse de chaque maille en un
`float4` lu par les passes suivantes, et une maille qui ne peut pas se vider (h ≥ dt·O) prend α = 1 sans lire ses
voisines. dam_break 250×250 passe de 29 à 46 µs/pas en release avec 4 passes.

## Nappe et infiltration

Le modèle souterrain est celui de wolfgpu : un aquifère libre sous la surface, avec un échange vertical de type
Darcy et un écoulement latéral de Boussinesq. Le code est dans `shaders/solver/groundwater.slang`. On l'active
avec un bloc `"groundwater"` dans le JSON du cas :

```jsonc
"groundwater": { "kv": 2e-3, "kh": 1e-2, "sy": 0.3, "depth": 2.0, "table_depth": 1.5 }
```

Chaque champ est un nombre, un fichier `.npy` relatif à `assets/`, ou le nom d'un tableau du npz du cas. Les
champs disponibles :

- `kv` : conductivité verticale (m/s) ;
- `kh` : conductivité horizontale (m/s) ;
- `sy` : porosité de drainage ;
- substratum imperméable : `depth` (épaisseur de sol) ou `z_sub` (cote) ;
- état initial de la nappe : `h0` (épaisseur saturée), `head` (cote) ou `table_depth` (profondeur sous le sol).

Les cas importés enregistrent ces champs dans leur npz (`gw_*`).

- **Échange vertical.** La nappe a pour cote H = z_sub + h_gw et stocke Sy·h_gw. Une même lame d par pas est
  retirée d'une couche et ajoutée à l'autre, plafonnée par ce que chaque côté peut céder ou
  recevoir :
  - infiltration si zb + h > H : d = min(kv·h/(h_gw + 1 m)·dt, h, Sy·(zb − H)). Le taux croît avec la lame
    d'eau en surface et décroît quand la nappe se remplit (pas encore de Green-Ampt) ;
  - exfiltration si H > zb + h : d = −min(kv·dt, Sy·(H − zb)).

  Ce transfert direct conserve la masse exactement, sans passer par le limiteur. Il évite ainsi le trou de
  masse possible de wolfgpu, où l'alpha de la nappe réduit une exfiltration déjà créditée à la surface.
- **Convention wolfgpu** (`"datum": "wolfgpu"`), pour les comparaisons. Elle reproduit le modèle d'origine
  (Sy = 1, aquifère posé sur le lit, sans limite de stockage). Le test vertical y compare zb + h à h_gw lui-même,
  alors que la charge latérale vaut zb + h_gw : c'est l'incohérence de datum du code wolfgpu, conservée telle
  quelle.
- **Écoulement latéral.** À travers chaque face, le débit vaut T·ΔH/d, avec la transmissivité T = kh·h_gw de la
  maille amont. Le schéma est explicite, et le pas de temps est limité à dt ≤ Sy / (4·T·(1/dx² + 1/dy²)),
  calculé sur GPU comme s_max. La positivité est assurée par un facteur de sortie par maille (une passe du
  limiteur de vidange).
- **Grille complète.** Les kernels de la nappe (`gw_alpha`, `gw_lateral`, `vertical_step`) tournent sur toute la
  grille après le dernier étage, car la nappe coule aussi sous les tuiles sèches.
  - Une exfiltration qui mouille une tuile sèche la marque pour la liste active du pas suivant.
  - Les résultats restent identiques au bit près avec `--all-tiles` (vérifié par `validate.py --all`).
- **Somme exacte.** h_gw est accumulé en virgule fixe 64 bits (2^-40 m), comme l'horloge, et `gw.x` n'en garde
  qu'une copie float. En float32, un échange lent (1e-8 m par pas) est sous l'ulp d'une nappe épaisse (3e-8 m à
  0,3 m) : il disparaissait, et l'eau quittait la surface sans arriver dans la nappe. Une somme compensée
  (Kahan) ne tient pas, car `precise` n'est pas transmis au SPIR-V.
  - La nappe reçoit exactement la lame perdue par la surface, h − fl(h − d).
  - Les cumuls des chronologies sont eux aussi en virgule fixe (signée).
  - Bilan surface + nappe : de 1e-8 à 4e-7 sur les cas de test, au lieu de 1,6e-4 en float.

Affichage :

- **Carte au sol** : les grandeurs de la nappe colorent le **sol** (profondeur de la nappe, saturation, échange
  en palette divergente bleu = exfiltration / brun = infiltration, lame infiltrée nette, cote piézométrique).
  Dans ces modes, l'eau devient presque transparente pour laisser lire la carte.
- **Isopièzes** sur le terrain (`--gw-contours <m>` ou panneau Affichage).
- **Nappe en transparence** (`--gw-xray`) : la surface de la nappe est vue à travers le sol, en bleu clair près
  de la surface et foncé en profondeur, sans test de profondeur.
- **Lignes de courant de la nappe** (`--gw-lines`) : les lignes de courant suivent la vitesse de pore
  −kh·∇H/Sy, en bleu sur le sol, au lieu de l'eau de surface.
- **Boîte de coupe** (`--crop i0,j0,i1,j1` ou panneau Affichage > Coupe) : seul un rectangle de mailles est
  dessiné. Ses parois forment un bloc-diagramme de la colonne de sol, peint par pixel : sol non saturé, sol
  saturé en bleu (tirets), ligne de la nappe, substratum gris. Le socle descend sous le substratum le plus bas.
- **Profil** (Courbes > Profil, `--show-profile`) : coupe verticale le long d'une ligne j, d'une colonne i ou de
  la sonde 1 à la sonde 2. Elle montre le substratum, le sol saturé, le sol non saturé, l'eau, le terrain et la
  cote de la nappe. Les valeurs sont lues sur le GPU à chaque image (kernel `profile`, 2048 points au plus).
- **Bilan** : volume de la nappe, volumes infiltrés et exfiltrés, et une ligne « Bilan » qui vérifie que
  surface + nappe − apports reste constant.
- **Courbes** : les sondes deviennent des piézomètres, et l'onglet « Nappe » trace les volumes.

Comparaison avec wolfgpu : `tools/gw_compare_wolfgpu.py --wolfgpu <repo>`, à lancer dans l'environnement Python
de wolfgpu, ou `validate.py --all --wolfgpu <repo>`.

- **Scénario** : un même problème lancé dans les deux codes jusqu'au même temps physique (`--time`). Une lame
  d'eau couvre un dôme de nappe, en convention wolfgpu.
- **Concordance de h_gw** : 3e-5 en L1 relatif, 4e-5 m au plus. L'écart s'explique par le décalage de 0,08 s
  entre les temps finaux.
- **Conservation** : SWE-Vulkan conserve exactement surface + nappe, wolfgpu dérive de 1e-6.

Cas de test :

- `gw_dam_break` : rupture de barrage sur sol perméable ;
- `gw_seepage` : retenue posée avec `--water` sur un sol mince, qui sature et ressort en résurgence ;
- `gw_wolfgpu` : même rupture de barrage, en convention wolfgpu.

### Pluie et infiltration par zones (chronologies wolfgpu)

C'est le pendant de `add_infiltration` / `add_rain` de wolfgpu. Une carte de zones par cellule (1 = première zone,
0 = aucune) est associée à une table temporelle des débits par zone, en m³/s pour la zone entière, répartis
uniformément sur ses mailles. Un débit positif ajoute de l'eau en surface, et une infiltration négative n'en
retire au plus que ce qui est présent. Comme dans wolfgpu, il n'y a rien avant la première ligne et la dernière
reste constante, avec une interpolation linéaire entre les lignes si `lerp` est activé.

```jsonc
"rain": { "zones": [[0, 0, 124, 249], [125, 0, 249, 249]], "unit": "mm/h", "lerp": true,
          "chronology": [[0, [3000, 0]], [20, [3000, 6000]], [40, [0, 0]]] },
"infiltration": { "zones": "all", "chronology": [[10, [-0.002]]] }
```

- **`zones`** : `"all"`, une liste de rectangles `[i0, j0, i1, j1]` (zone k = rectangle k), un `.npy` ou un
  tableau du npz.
- **`unit: "mm/h"`** : facilité pour la pluie, convertie en m³/s avec l'aire de chaque zone.
- **Une même maille peut être à la fois dans une zone de pluie et dans une zone d'infiltration**, comme dans
  wolfgpu.
- **Calcul des débits** : `finalize_dt` les cherche une fois par pas, au temps de début du pas, puis le kernel
  vertical `vertical_step` les applique sur toute la grille, avant l'échange avec la nappe.
- **Tuiles** : la pluie sur une tuile sèche l'active au pas suivant, et les résultats restent identiques au bit
  près avec `--all-tiles`.

Édition interactive (Simulation > outils d'édition > Pluie / infiltration) :

- choisir pluie ou infiltration ;
- « Zone = sélection » ou « Zone = domaine » crée une zone avec un débit constant ;
- la table temps × zones s'édite (Entrée pour valider) : « + ligne », suppression d'une ligne (x) ou d'une
  zone (x de l'en-tête), interpolation linéaire ;
- valeurs en mm/h ou en m³/s par zone, au choix ;
- les zones sont hachurées sur le terrain (pluie en bleu, infiltration en brun, liseré aux frontières) quand
  « Afficher les zones » est coché.

En ligne de commande : `--rain i0,j0,i1,j1,mm_h` et `--infiltration i0,j0,i1,j1,mm_h` (débit constant dès t = 0,
options répétables).

Affichage :

- le mode « Pluie cumulée » (mm) colore le sol ;
- le Bilan donne la pluie et l'apport net, et la ligne « Bilan » vérifie que surface + nappe − apports reste
  constant ;
- l'onglet « Nappe » (ou « Pluie ») de Courbes trace la pluie cumulée.

Le cas de test est `rain_infiltration` : pluie sur deux zones, extraction par infiltration, et nappe. La pluie
attendue y est de 2,9167 m³, contrôlée par `validate.py --all`.

`tools/groundwater_check.py` contrôle la positivité et le bilan surface + nappe − apports d'un `--dump`, qui
contient alors aussi `gw.npy` et/ou `sources.npy`.

## Validation contre le Python

```bat
python tools\validate.py --all        :: 3 schémas sur dam_break vs Taichi, meuse, vesdre, débit, limiteur
python tools\validate.py dam_break ssprk3 200
python tools\lake_at_rest.py          :: équilibre du lac au repos (île sèche)
python tools\positivity_check.py out\run --conserve   :: h >= 0, aucune troncature, volume conservé
```

`tools/dump_reference.py` exécute le `main.py` d'origine sans le modifier et écrit le même format de dump.
Avec `--shu-osher`, il exécute une copie du projet (`out/_swe_shu_osher`) dont la boucle Runge-Kutta est réécrite
en forme de Shu-Osher, l'ordre d'opérations du C++ : la forme de Butcher de l'original n'en diffère que par
l'arrondi, mais le choc l'amplifie (à 10 pas déjà 1 000 cellules au-dessus de 1e-5, dt différent de 6e-6) ;
Euler, identique dans les deux formes, reste comparé au Python d'origine. Les comparaisons tournent avec le limiteur
de vidange coupé (`--drain 0`), le Python tronquant h < 0.

Résultats (M1, RTX 5070 Ti) :

- Euler 50 pas : écart relatif max < 1e-6 sur h, hu, hv ; t et dt identiques.
- ssprk3 (200 pas) : écart max ~2e-6.
- ssprk2 (200 pas) : L1 relatif ~1e-6, mais quelques dizaines de cellules au front de choc / sec-mouillé
  jusqu'à ~1e-3 (une vingtaine en h, 73 en hu). **Taichi CPU contre Taichi Vulkan montre exactement le même motif**
  (6 cellules en h, 38 en hu) : c'est l'amplification d'arrondi au front, pas un défaut du portage. D'où le critère :
  L1 ≤ 1e-5 et au plus 128 cellules au-dessus de 1e-5.
- Lac au repos (1000 pas) : |eta − 0.5| ≤ 1e-6 ; courants parasites ~1e-5, identiques au Python
  (même cellule, même amplitude : arrondi f32 de l'équilibre hydrostatique).
- Meuse (100 pas, débit + niveau + Manning) : `zb` identique, écart max ~3e-6.
- Vesdre (100 pas, 0,48 M mailles) : `zb` identique, L1 ~1e-6, ~100 cellules de front au-dessus de 1e-5,
  comme Taichi CPU contre Taichi Vulkan (94). Tolérance de front : max(128, 0,05 % des mailles).
- Limiteur de vidange : `drain_test` (Euler, CFL 0,9, 1000 pas) tronque 10 867 fois jusqu'à 0,48 m sans limiteur
  (+6,5e-4 de volume) ; avec 1, 4 ou 8 passes : h ≥ 0 partout, aucune troncature, volume conservé à 3e-8 près
  (arrondi f32 des mises à jour). Idem dam_break, lake_at_rest, vesdre.
