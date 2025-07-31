# SDGSAT-composites
Création de composites (valeur moyenne) à partir des images pré-traitées (radiométriquement et géométriquement) SDGSAT-1.

## Utilisation
```
git clone https://github.com/SCO-ORENOS/SDGSA-composite
cd SDGSAT-composite

# Modifier environment.yml si besoin de changer le nom de l'env
conda env create -f environment.yml
conda activate sdgsat-composite

# Modifier config.yml avant de lancer le script
python composite.py --config_file config.yml
```
