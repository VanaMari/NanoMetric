
# Deep Learning-Based Segmentation of Nanomaterial Images Acquired via Scanning Electron Microscopy

Os arquivos neste repositório acompanham a seguinte publicação:

**Donato, I.M.L.; Marques, F. D.; Archanjo, B. S.; Lopes, F.J.P.; Lunz, J. N. and Girald, G.A.**  
*Deep Learning-Based Segmentation of Nanomaterial Images Acquired via Scanning Electron Microscopy.*  
Revista de Informática Teórica e Aplicada, 2024.  

Se você usar os arquivos ou o código em seu próprio trabalho, cite o artigo acima.

Salvo indicação em contrário, todos os arquivos nesta página são fornecidos sob a licença MIT.

---

## Descrição

Este repositório contém o código e os dados utilizados no artigo que investiga a aplicação de técnicas de aprendizado profundo para segmentar imagens de nanomateriais obtidas por microscopia eletrônica de varredura (MEV). O estudo aborda a segmentação de nanopartículas de óxido de zinco (ZnO) e nanofolhas de óxido de grafeno (GO) usando a arquitetura U-Net. Além disso, utiliza estratégias de aprendizado com poucos dados (Few-Shot Learning) para realizar transferência de aprendizado entre os dois tipos de imagens.

## Organização do Repositório

- `data/`: Contém as imagens de entrada e os respectivos padrões-ouro utilizados para treinamento, validação e teste.
- `models/`: Inclui os pesos dos modelos treinados (U-Net) e os scripts de configuração.
- `notebooks/`: Jupyter Notebooks para replicar os experimentos do artigo.
- `scripts/`: Scripts Python para preprocessamento, treinamento, validação e avaliação dos modelos.
- `results/`: Contém os resultados das métricas e análises realizadas.

## Instruções de Uso

### Requisitos

- **Linguagem de Programação**: Python 3.8 ou superior.
- **Dependências**: Listadas no arquivo `requirements.txt`.
- **Plataforma de Treinamento**: Recomendamos o uso do Google Colab ou uma GPU com suporte para CUDA.

### Passos para Reproduzir os Experimentos

1. **Instale as Dependências**  
   Execute:  
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare os Dados**  
   Certifique-se de que as imagens de ZnO e GO estão organizadas no formato esperado no diretório `data/`.

3. **Treine o Modelo**  
   Para treinar o modelo U-Net com as imagens de ZnO:  
   ```bash
   python scripts/train_unet.py --dataset znO
   ```

4. **Realize o Fine-Tuning**  
   Para ajustar o modelo com imagens de GO, execute:  
   ```bash
   python scripts/fine_tune.py --dataset go
   ```

5. **Avalie o Modelo**  
   Use o script de avaliação para gerar métricas como IoU, F1-Score, Precisão e Recall:  
   ```bash
   python scripts/evaluate.py --model trained_model.h5
   ```

## Principais Contribuições do Estudo

1. **Banco de Dados de Alta Resolução**: Contendo pares de imagens originais e padrões-ouro, disponíveis para a comunidade científica.
2. **Estratégias de Segmentação Otimizadas**: Investigação de diferentes técnicas de recorte e transferência de aprendizado.
3. **Resultados Promissores**: Métricas superiores a 95% para ambos os tipos de imagens.

## Licença

Este repositório é distribuído sob a licença MIT. Consulte o arquivo `LICENSE` para obter mais detalhes.

---

## Referências

Para informações detalhadas, consulte o artigo completo na [Revista de Informática Teórica e Aplicada](http://dx.doi.org/10.22456/2175-2745.XXXX).
