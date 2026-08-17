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

### Executar com duplo clique (Ubuntu)

Alternativamente, para evitar usar o terminal, é possível configurar o arquivo `run_gui.sh` para rodar com duplo clique do mouse. 

Para abrir a GUI clicando duas vezes em `run_gui.sh` no gerenciador de arquivos, é preciso
marcar o arquivo como executável **uma vez**:

1. Abra a pasta `FeretImageProcessing` no Gerenciador de arquivos (Nautilus).
2. Clique com o **botão direito** em `run_gui.sh`.
3. Escolha **Propriedades**.
4. Vá à aba **Permissões**.
5. Marque **Permitir executar arquivo como programa** (ou **Executar como programa**).
6. Feche a janela de propriedades.

Depois disso, um **duplo clique** em `run_gui.sh` deve executar o script (ele ativa o `.venv`
e abre a GUI).

Se o Ubuntu perguntar o que fazer com o arquivo, escolha **Executar** ou **Executar no
Terminal** — a GUI precisa de terminal para mostrar o log do pipeline. Se abrir só o editor
de texto, volte às Propriedades e confira se a opção de executar está marcada; em alguns
sistemas também é preciso definir o comportamento padrão de arquivos `.sh` como “Executar”
nas preferências do gerenciador de arquivos.

Equivalente pelo terminal (mesma permissão):

```bash
chmod +x run_gui.sh
./run_gui.sh
```

### Uso da GUI

1. **Pasta da campanha** — pasta da campanha, com layout `campanha/data/frames/imagens`
   (cada data contém subpastas de frames, por exemplo `Config 01` ou `Basler_*_frames`).
2. **Pasta base de saída** — normalmente `outputs/`; o pipeline grava em
   `outputs/<campanha>/<data>/` (`roi_crops`, Feret CSV, `.npz` e previews, conforme as opções).
3. Ajuste os parâmetros de segmentação, Feret e ROI se necessário e clique em **Executar pipeline**.

Por padrão, se a pasta de saída de uma data já existir, só são processadas imagens que ainda
não têm saída completa (ROI, `.npz` e side-by-side, quando habilitados).

Para verificar o que faz cada parâmetro, cheque o arquivo `docs/parametros.html`.

> **Nota:** o processamento de uma campanha inteira pode demorar bastante.

## Próximas execuções

Depois da instalação inicial, para rodar o programa, há 2 alternativas:

### Executar com duplo clique 

Ver seção "Executar com duplo clique (Ubuntu)" acima.

### Rodar pelo terminal
Abrir o terminal na pasta do projeto e rodar:

```bash
bash run_gui.sh
```

Ou ativar o `.venv` e chamar `python3 pipeline_gui.py`, como acima.
