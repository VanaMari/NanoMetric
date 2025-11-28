# 📁 NanoMetric

Este repositório contém os arquivos, bancos de dados e modelos utilizados na pesquisa de doutorado voltada à **segmentação de imagens de microscopia eletrônica de varredura (MEV)** de **óxido de zinco (ZnO)** e **óxido de grafeno (GO)** por meio de redes neurais convolucionais do tipo **U-Net**, com ênfase em técnicas de **aprendizado com poucos exemplos (Few-Shot Learning)** e **ajuste fino (Fine-Tuning)**.

---

## 📘 Estrutura do Repositório

### 🧩 1. Arquitetura da Rede

**Pasta:** `Arquitetura_U_Net/`  
- **Arquivo:** `unet_model.py`  
  - Contém a implementação completa da arquitetura U-Net utilizada nos experimentos, configurada para imagens de entrada com 3 canais (RGB) e saída binária (máscara de segmentação).  
  - Inclui funções de compilação, definição de camadas e parâmetros de treinamento.

---

### 🧠 2. Banco de Dados

**Pasta:** `Banco_de_Dados/`  
Contém os conjuntos de imagens originais de óxido de zinco (ZnO) e óxido de grafeno (GO). Estas imagens incluem a barra de informações sobre a aquisição das mesmas.

- **Subpasta:** `GO/`  
  - 8 imagens de óxido de grafeno (GO), obtidas por microscopia eletrônica de varredura (MEV).  

- **Subpasta:** `ZnO/`  
  - 97 imagens de óxido de zinco (ZnO), também obtidas por MEV.

---

### ⚙️ 3. Treinamentos — ZnO

**Pasta:** `Treinamentos/ZnO/` 

- **Subpasta:** `Original_ZnO/`  
  - Imagens originais de ZnO utilizadas como entrada no treinamento inicial.

- **Subpasta:** `Padrao_Ouro_ZnO/`  
  - Máscaras binárias, segmentadas, correspondentes às imagens de ZnO, usadas como ground truth.

- **Subpasta:** `pesos/`  
  Contém os pesos dos modelos U-Net treinados com o conjunto de ZnO.  
  - **Estrategia_de_RecorteI_MI/**  
  - **Estrategia_de_RecorteI_MII/**  
  (Cada subpasta refere-se a uma estratégia de recorte distinta aplicada às imagens durante o treinamento.)

---

### ⚙️ 4. Treinamentos — GO

**Pasta:** `Treinamentos/GO/`  

- **Subpasta:** `Original_GO/`  
  - Imagens originais utilizadas como entrada para treinamento da U-Net.

- **Subpasta:** `Padrao_ouro_GO/`  
  - Máscaras binárias de referência (ground truth).

- **Subpasta:** `pesos_AjusteFino/`  
  Contém os pesos gerados durante o processo de *fine-tuning* dos modelos previamente treinados em ZnO.  
  - **Estrategia_de_RecorteII_MI/**  
  - **Estrategia_de_RecorteI_MII/**  
  (As diferentes estratégias indicam variações no método de recorte e no modelo base utilizado.)

- **Subpasta:** `pesos_FromScratch/`  
  Pesos resultantes do treinamento From scratch, apenas com as imagens de GO.  
  - **Estrategia_de_RecorteI_MI/**  
  - **Estrategia_de_RecorteI_MII/**
---

### 🧮 5. NanoMetric — Medição e Análise

**Pasta:** `NanoMetric/`  
- **Arquivo:** `NanoMetric.py`  
  - Executa a rotina completa de segmentação automática (U-Net) e medição de partículas.  
  - Calcula diâmetros de Feret (máx., mín. e médio), área, perímetro, circularidade e gera relatórios em `.csv` e imagens segmentadas.  
  - Também exporta metadados com as configurações de calibração e versões das bibliotecas utilizadas.

---

## 🧪 Como Reproduzir os Experimentos

### 1️⃣ Requisitos

| Biblioteca | Versão recomendada |
|-------------|--------------------|
| Python | 3.8 |
| NumPy | 1.25.2 |
| Matplotlib | 3.7.1 |
| TensorFlow | 2.15.0 |
| Pandas | 2.0.3 |
| Scikit-Learn | 1.2.2 |
| OpenCV | ≥ 4.7 |

> 💡 É recomendado executar no **Google Colab** ou em ambiente local (Anaconda/Spyder) com GPU disponível.

---

### 2️⃣ Estrutura necessária

Coloque os seguintes diretórios e arquivos dentro da pasta principal `TESE_Ivania_01.12.25/`:

```
├── Arquitetura_U_Net/
│   └── unet_model.py
├── Banco_de_Dados/
│   ├── ZnO/
│   │   └── *.tif, *.png, ...
│   └── GO/
│       └── *.tif, *.png, ...
├── Treinamentos/
│   ├── ZnO/...
│   └── GO/...
└── NanoMetric/
    └── NanoMetric.py
```

---

### 3️⃣ Segmentação automática

O script **`NanoMetric.py`** carrega o modelo U-Net definido em `unet_model.py`, aplica-o às micrografias e gera máscaras binárias de segmentação.

No início do arquivo, ajuste o caminho do modelo pré-treinado:

```python
UNET_MODEL_PATH = 'caminho/para/seus/pesos_ZnO/mII.weights.h5'
```

E defina o diretório contendo as imagens a processar:

```python
CAMINHO_PASTA = r'/content/Banco_de_Dados/ZnO'
```

---

### 4️⃣ Execução

No **Colab**, execute:

```python
!python /content/TESE_Ivania_01.12.25/NanoMetric/NanoMetric.py
```

ou, no **Spyder**/**terminal local**:

```bash
python "C:\Users\...\TESE_Ivania_01.12.25\NanoMetric\NanoMetric.py"
```

Durante a execução:
- As imagens são segmentadas pela U-Net;
- As partículas são identificadas por contorno;
- São calculados **diâmetros de Feret máximo, mínimo e médio**, área, perímetro e circularidade;
- Resultados e estatísticas são exportados automaticamente.

---

### 5️⃣ Saídas geradas

| Arquivo | Descrição |
|----------|------------|
| `segmentadas/` | Máscaras binárias segmentadas pela U-Net |
| `resultados_particulas.csv` | Medidas individuais (área, circularidade, Feret máx./mín./médio etc.) |
| `resultados_estatisticas.csv` | Estatísticas globais por imagem (média, desvio padrão, n de partículas) |
| `metadados_pipeline.csv` | Informações de calibração, parâmetros e versões do ambiente |

---

### 6️⃣ Parâmetros ajustáveis

| Parâmetro | Função |
|------------|--------|
| `PIXEL_TO_MICROMETER` | Fator de calibração (µm/pixel) |
| `SIZE_MIN`, `SIZE_MAX` | Limites de área (µm²) |
| `CIRCULARITY_MIN`, `CIRCULARITY_MAX` | Filtro por circularidade |
| `ANGLE_STEP` | Resolução angular do cálculo de Feret |
| `CROP_SIZE`, `OVERLAP` | Dimensão e sobreposição dos recortes da U-Net |
| `INCLUDE_CUTOFF_PARTICLES` | Incluir partículas cortadas na borda |
| `FILL_HOLES` | Preencher buracos internos nas partículas |

---

### 7️⃣ Protocolo de medição resumido

1. **Pré-processamento:** leitura e normalização das imagens.  
2. **Segmentação (U-Net):** geração de máscaras binárias.  
3. **Detecção de contornos:** identificação das partículas.  
4. **Cálculo geométrico:** área, perímetro, circularidade e diâmetros de Feret (máx., mín., médio).  
5. **Filtragem:** exclusão por área ou circularidade.  
6. **Exportação:** resultados consolidados em `.csv` e imagens segmentadas.

---

## 📄 Resumo Geral

| Categoria | Descrição | Conteúdo |
|------------|------------|-----------|
| **Arquitetura** | Modelo U-Net utilizado nos experimentos | `unet_model.py` |
| **Banco de Dados** | Imagens de ZnO (97) e GO (8) | `Banco_de_Dados/` |
| **Treinamentos ZnO** | Modelos base treinados com ZnO | `Treinamentos/ZnO/` |
| **Treinamentos GO** | Fine-tuning e treinamentos do From scratch | `Treinamentos/GO/` |
| **Medições** | Análises dimensionais via NanoMetric.py | `NanoMetric/` |

---

## 🧾 Observações

- Todos os experimentos foram realizados com **imagens 256×256 px**, obtidas por recortes das micrografias originais.  
- Os pesos armazenados correspondem a diferentes **estratégias de recorte** e **modelos (MI e MII)** empregados na análise comparativa.  
- O arquivo `NanoMetric.py` pode ser utilizado para reconstruir e carregar qualquer um dos modelos cujos pesos estão disponíveis nas pastas correspondentes.

## 🧾 Referências

- Para informações detalhadas, consulte o artigo completo na [Revista de Informática Teórica e Aplicada](http://dx.doi.org/10.22456/2175-2745.XXXX).
