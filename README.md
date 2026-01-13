# ABBA_LSTM

**ABBA_LSTM** est un projet Python combinant la **représentation symbolique ABBA** (Adaptive Brownian Bridge-based Aggregation) avec un modèle **LSTM** pour la prédiciton de séries temporelles.

L’objectif est de réduire la complexité des séries temporelles grâce à ABBA, puis d’exploiter les dépendances temporelles via un réseau de neurones récurrent de type LSTM.

---

## 🚀 Fonctionnalités

- **Encodage ABBA** des séries temporelles
- **Apprentissage séquentiel** avec un modèle LSTM
- **Notebook interactif** pour l’exploration et l’expérimentation
- **Export des résultats** au format CSV

---

## 🔧 Installation

Clone le dépôt :

```bash
git clone https://github.com/VSmague/ABBA_LSTM.git
cd ABBA_LSTM
```

Créer un environnement virtuel (recommandé) :

```
python3 -m venv venv
source venv/bin/activate  # Windows : venv\Scripts\activate
```

Installer les dépendances :

```
pip install -r requirements.txt
```

## Référence

ABBA – Adaptive Brownian Bridge-based Aggregation [https://arxiv.org/abs/2003.12469]
