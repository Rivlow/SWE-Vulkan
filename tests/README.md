# Tests hydrauliques (pytest)

```bat
python -m pip install -r requirements-dev.txt
python -m pytest tests -q                    :: build\release\bin\swe.exe
python -m pytest tests -q --preset debug
python -m pytest tests -q -m "not slow"
python -m pytest tests -q --wolfgpu C:\...\wolfgpu\Repository   :: profils sauvegardés de wolfgpu (détecté par défaut)
```

Chaque test écrit son cas dans `out/tests/<test>/assets` (type `imported` : JSON + npz, `tests/swetest.py`
`Case`), lance `swe --headless --no-validation --dump ... [--record ...]` (fixture `run`, `tests/conftest.py`) et lit
le dump et l'enregistrement (`tools/swerec.py`, `tools/swepost.py`). Le journal de chaque exécution est à côté.

## Masse d'abord

`swetest.assert_mass` est appelée par presque tous les tests :

- h ≥ 0, état fini, aucune maille tronquée à 0 (une troncature crée de la masse) ;
- **fermeture** : V − V0 − apports = 0, les apports étant les volumes exacts mesurés par le solveur à travers chaque
  bord, groupe de faces et zone (cumul entier 64 bits), plus pluie / infiltration / nappe (`meta.json`
  `added_total`). Domaines fermés : 1e-6 relatif (observé 1e-8) ; domaines ouverts : 4 √N ulp float32 (arrondi des
  mises à jour, N pas) ;
- la même fermeture **à chaque enregistrement** (volumes des bords dans le `.swerec`) ;
- `expected_added` : ce que les conditions doivent apporter (Q t pour un débit imposé), à 1e-5 près.

## Correspondance avec wolfgpu (`C:\Users\lucas\Dev\wolfgpu\Repository\tests`)

| Ici | wolfgpu | Critère ici |
|---|---|---|
| `test_mass_conservation` (fermés, sources, débit, faces, tuiles, nappe, pluie) | `test_cube_drop`, `test_small_splash`, `test_sine_ground_sine_water_3lanes`, `test_infiltration_over_time*`, `test_injection` | fermeture exacte, tuiles actives = grille complète au bit près |
| `test_lake_at_rest` | `test_still_water_rocky_bottom` (+ `_fens`, `_alpine`), `test_lakeatrest_rockybottom_*`, `test_v_shape_pool_and_no_flow`, `test_hmod_hold_water` | **au bit près** (wolfgpu : Σ\|q\| < 2e-4) |
| `test_uniform_flow` | `test_waterline_along_X_*`, `test_still_flow_*`, `test_waterline_along_X_3lines_rough` | 1e-6 ; hauteur normale 3e-4 |
| `test_analytical` (Ritter, Stoker, SWASHES, M1 / M2) | `Analysis_linear_reconstruction.py`, `Analysis_flux_formulation.py` (sans assertion) | L1, convergence en dx, position des fronts / ressaut |
| `test_boundary_conditions` | `test_h_symmetry_*`, `test_hmod_line_*` | symétrie 1e-5, volumes par bord |
| `test_bridges` | `test_bridges.py` (charge seulement, résultats sauvegardés) | repos exact, continuité, pente de frottement en charge, Bernoulli, seuil, surverse, profils wolfgpu |
| `test_recording` | — | temps, f16, `--record-dir`, débit de section |

## Ce que les tests ont révélé (et ce qui a été corrigé)

- Courants parasites du lac au repos (2e-4 m²/s sur fond rocheux) : pression écrite en écart, η reconstruit → exact.
- Perte de masse des incréments sous l'ulp (1e-4 en 30 000 pas) : somme exacte et report.
- Limiteur de Venkatakrishnan à 0,75 sur données lisses et seuil de film mince 1 mm : front de Ritter 3,5 m en
  retard → corrigés (`--reference` garde l'ancien comportement pour `tools/validate.py`).
- Euler + MUSCL instable : documenté, Euler n'est plus comparé au Python au-delà de 50 pas.
- Conditions faibles de hauteur / niveau : la maille de bord s'écarte de quelques % de la valeur imposée (fantôme),
  une sortie libre n'est pas un contrôle en fluvial : documenté dans les tests concernés.
