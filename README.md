# FeretImageProcessing

Código que faz a **segmentação** e o **cálculo dos diâmetros de Feret** das imagens, além da
**extração de ROIs** (recortes de cada componente segmentado).

O projeto e os pacotes instalados ficam em `Documentos > Programas`.

## Interface gráfica (GUI)

Na pasta `FeretImageProcessing`, abra um terminal e rode:

```bash
bash run_gui.sh
```

O script ativa o ambiente virtual (`.venv`) e abre a interface PyQt5 do pipeline
(segmentação, Feret e extração de ROIs). Escolha a pasta da **campanha**
(`campanha/data/frames/imagens`); as saídas vão para `outputs/<campanha>/<data>/`.

Se o `.venv` ainda não existir:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

> **Nota:** este processo pode demorar bastante.
