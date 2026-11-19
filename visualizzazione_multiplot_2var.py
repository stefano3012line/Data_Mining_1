"""
Visualizzazione a matrice di scatter plot bivariati (scatterplot matrix / pairplot)
per un dataset con un numero generico di attributi, usando solo matplotlib.

Uso tipico:
    import pandas as pd
    df = pd.read_csv("dati.csv")
    scatter_matrix(df, diag="hist")

Funziona sia con un pandas DataFrame sia con un array numpy (N_samples, N_features).
"""

import numpy as np
import matplotlib.pyplot as plt

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False


def scatter_matrix(
    data,
    labels=None,
    classes=None,
    diag="hist",          # "hist", "kde" (richiede scipy), oppure None
    figsize=None,
    point_size=12,
    alpha=0.6,
    cmap="red",            # colormap per scatter se classes è None
    corner=False,          # True -> mostra solo il triangolo inferiore
    save_path=None,
    dpi=150,
):
    """
    Disegna una matrice NxN di scatter plot bivariati per N attributi.

    Parametri
    ---------
    data : array-like (n_samples, n_features) oppure pandas.DataFrame
        Il dataset. Ogni colonna è un attributo.
    labels : list of str, opzionale
        Nomi degli attributi. Se `data` è un DataFrame e `labels` è None,
        vengono usati i nomi delle colonne.
    classes : array-like (n_samples,), opzionale
        Etichette di classe/categoria per colorare i punti (es. labels di
        un dataset di classificazione). Se fornito, i punti vengono colorati
        in base alla classe e viene aggiunta una legenda.
    diag : {"hist", "kde", None}
        Cosa disegnare sulla diagonale principale.
    figsize : tuple, opzionale
        Dimensione della figura. Default: scala con il numero di attributi.
    point_size : float
        Dimensione dei marker negli scatter.
    alpha : float
        Trasparenza dei punti (utile con molti dati sovrapposti).
    cmap : str
        Colormap usata se `classes` è None e si vuole comunque una mappa
        continua (es. per colorare in base a un attributo esterno).
    corner : bool
        Se True, disegna solo il triangolo inferiore della matrice
        (utile quando N è grande, dimezza il numero di pannelli).
    save_path : str, opzionale
        Se fornito, salva la figura su file (png, pdf, svg, ...).
    dpi : int
        Risoluzione di salvataggio/visualizzazione.

    Ritorna
    -------
    fig, axes : la figura e l'array di assi matplotlib, per ulteriori
        personalizzazioni dopo la chiamata.
    """

    # --- Normalizza l'input in un array numpy (n_samples, n_features) ---
    if _HAS_PANDAS and isinstance(data, pd.DataFrame):
        if labels is None:
            labels = list(data.columns)
        X = data.to_numpy(dtype=float)
    else:
        X = np.asarray(data, dtype=float)
        if labels is None:
            labels = [f"x{i}" for i in range(X.shape[1])]

    n_samples, n_features = X.shape
    assert len(labels) == n_features, "labels deve avere un elemento per colonna"

    # --- Gestione classi/colori ---
    if classes is not None:
        classes = np.asarray(classes)
        unique_classes = np.unique(classes)
        colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(unique_classes)))
        class_to_color = {c: colors[i] for i, c in enumerate(unique_classes)}
        point_colors = np.array([class_to_color[c] for c in classes])
    else:
        point_colors = "red"

    # --- Figura e assi ---
    if figsize is None:
        side = max(2.0, min(3.0, 18.0 / n_features))
        figsize = (side * n_features, side * n_features)

    fig, axes = plt.subplots(
        n_features, n_features, figsize=figsize, squeeze=False
    )

    for i in range(n_features):      # riga -> asse y
        for j in range(n_features):  # colonna -> asse x
            ax = axes[i, j]

            if corner and j > i:
                ax.axis("off")
                continue

            if i == j:
                # --- Diagonale: distribuzione marginale ---
                if diag == "hist":
                    ax.hist(X[:, j], bins=25, color="gray", alpha=0.8)
                elif diag == "kde":
                    from scipy.stats import gaussian_kde
                    xs = np.linspace(X[:, j].min(), X[:, j].max(), 200)
                    kde = gaussian_kde(X[:, j])
                    ax.plot(xs, kde(xs), color="gray")
                    ax.fill_between(xs, kde(xs), alpha=0.3, color="gray")
                else:
                    ax.axis("off")
            else:
                # --- Fuori diagonale: scatter bivariato ---
                ax.scatter(
                    X[:, j], X[:, i],
                    s=point_size, alpha=alpha,
                    c=point_colors, cmap=cmap,
                    edgecolors="none",
                )

            # --- Etichette solo sui bordi della matrice ---
            if i == n_features - 1:
                ax.set_xlabel(labels[j], fontsize=9)
            else:
                ax.set_xticklabels([])

            if j == 0:
                ax.set_ylabel(labels[i], fontsize=9)
            else:
                ax.set_yticklabels([])

            ax.tick_params(axis="both", labelsize=7)

    # --- Legenda classi, se presenti ---
    if classes is not None:
        handles = [
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=class_to_color[c], markersize=7, label=str(c))
            for c in unique_classes
        ]
        fig.legend(handles=handles, loc="upper right", title="Classe",
                   bbox_to_anchor=(0.995, 0.995))

    fig.suptitle("Matrice di scatter plot bivariati", fontsize=13, y=1.01)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig, axes


if __name__ == "__main__":   
    "Questo è il pezzo da modificare per importare un dataset diverso e cambiare le opzioni di visualizzazione."
    df = pd.read_csv(r"iris (1).data", header=None)

    # L'ultima colonna è il nome della classe, le altre sono gli attributi
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    fig, axes = scatter_matrix(X, labels=["sepal length", "sepal width", "petal length", "petal width"], classes=y,diag="hist",figsize=(8,8))

    plt.show()