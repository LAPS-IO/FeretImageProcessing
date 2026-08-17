# FeretImageProcessing

Código que faz a **segmentação** e o **cálculo dos diâmetros de Feret** das imagens, além da
**extração de ROIs** (recortes de cada componente segmentado).

O projeto fica em `Documentos > Programas > FeretImageProcessing`.

## Requisitos

- Python 3.12 (testado com 3.12.2)
- Terminal com acesso à pasta do projeto

## Instalação (primeira vez)

Abra um terminal na pasta `FeretImageProcessing` e execute, na ordem:

### 1. Criar o ambiente virtual

```bash
cd ~/Documentos/Programas/FeretImageProcessing
python3 -m venv .venv
```

Isso cria a pasta `.venv` com um Python isolado só para este projeto.

### 2. Ativar o ambiente virtual

```bash
source .venv/bin/activate
```

O prompt do terminal deve passar a mostrar `(.venv)` no início da linha.

### 3. Instalar as dependências

```bash
pip install -r requirements.txt
```

Pacotes principais: `numpy`, `opencv-python`, `tqdm`, `PyQt5`.

## Executar a interface gráfica

Com o ambiente virtual **ativado** (`source .venv/bin/activate`):

```bash
python3 pipeline_gui.py
```

Ou, sem precisar ativar manualmente (o script faz isso por você):

```bash
bash run_gui.sh
```

### Uso da GUI

1. **Campaign folder** — pasta da campanha, com layout `campanha/data/frames/imagens`
   (cada data contém subpastas de frames, por exemplo `Config 01` ou `Basler_*_frames`).
2. **Output base** — normalmente `outputs/`; o pipeline grava em
   `outputs/<campanha>/<data>/` (`roi_crops`, Feret CSV, `.npz` e previews, conforme as opções).
3. Ajuste os parâmetros de segmentação, Feret e ROI se necessário e clique em **Run pipeline**.

Por padrão, se a pasta de saída de uma data já existir, só são processadas imagens que ainda
não têm saída completa (ROI, `.npz` e side-by-side, quando habilitados).

> **Nota:** o processamento de uma campanha inteira pode demorar bastante.

## Próximas execuções

Depois da instalação inicial, basta abrir o terminal na pasta do projeto e rodar:

```bash
bash run_gui.sh
```

Ou ativar o `.venv` e chamar `python3 pipeline_gui.py`, como acima.
