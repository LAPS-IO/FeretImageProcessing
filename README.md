# FeretImageProcessing

Código que faz a **segmentação** e o **cálculo dos diâmetros de Feret** das imagens, além da
**extração de ROIs** (recortes de cada componente segmentado).

O projeto e os pacotes instalados ficam em `Documentos > Programas`.

## Extração de ROIs

O código para extrair os ROIs já está incluso. Para usá-lo:

### 1. Restaurar as pastas `run`

É importante ter as pastas `run` dentro de `outputs`. Copie as pastas do backup
(`Elements > Backup_outputs_Feret > outputs`) de volta para:

```
Documentos > Programas > FeretImageProcessing > outputs
```

### 2. Conferir o caminho das imagens

Dentro de cada pasta `run{N}/run_metadata.txt`, verifique qual foi o `root`, isto é, de que dia
foram as imagens processadas.

Anote o **caminho atual** para a pasta das imagens. Ele pode ser diferente do que está em `root`,
pois o `root` contém o caminho original de antes do computador ter sido formatado.

### 3. Ativar o ambiente virtual

Abra um terminal na pasta `FeretImageProcessing` e rode:

```bash
source .venv/bin/activate
```

### 4. Rodar a extração

Ajuste `{caminho para a pasta de imagens}` e `{N}`:

```bash
python run_extraction_all_folders.py {caminho para a pasta de imagens} outputs/run{N}/ -v
```

Por exemplo, para extrair os ROIs de **2/12/25**:

```bash
python run_extraction_all_folders.py /home/laps/Documentos/Limpeza_Casco/Raw/04_Campanha_Dezembro_2025/02122025 outputs/run2/ -v
```

### 5. Resultado

Se funcionar, os ROIs serão salvos em:

```
FeretImageProcessing/outputs/run{N}/roi_crops
```

> **Nota:** este processo pode demorar bastante.
