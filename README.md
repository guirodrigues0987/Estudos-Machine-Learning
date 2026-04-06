# 🍄 Neuroevolução no Super Mario Bros com NEAT-Python

Este projeto implementa uma Inteligência Artificial capaz de aprender a jogar Super Mario Bros (NES) utilizando o algoritmo **NEAT** (NeuroEvolution of Augmenting Topologies). O foco principal é a otimização de topologias de redes neurais através de processos evolutivos.

---

## 🚀 Visão Geral do Projeto

Diferente de abordagens tradicionais de Deep Learning, este projeto utiliza **Algoritmos Genéticos** para evoluir a estrutura e os pesos de uma rede neural. 

### Principais Diferenciais Técnicos:
* **Eficiência de Hardware:** Desenvolvido para rodar em ambientes com recursos limitados (Testado em **Lubuntu/Linux**).
* **Persistência de Dados:** Implementação de um sistema customizado de *Checkpoints* para garantir a resiliência do treinamento contra falhas de sistema ou quedas de energia.
* **Abordagem de Engenharia:** Foco em métricas de *Fitness* baseadas em deslocamento horizontal e bônus de tempo, otimizando a velocidade de convergência.

---

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python 3.11
* **Ambiente:** [Gymnasium](https://gymnasium.farama.org/) (Interface para RL)
* **Emulador:** `gym-super-mario-bros`
* **Algoritmo:** `neat-python` (Neuroevolução)
* **Processamento:** OpenCV (Redimensionamento e escala de cinza dos frames)

---

## 📈 Resultados e Evolução

O treinamento é dividido em gerações de **50 indivíduos**. 
- **Geração 0:** Movimentos aleatórios.
- **Geração 20+:** Início de padrões de salto sobre obstáculos e inimigos.
- **Geração 100+:** Domínio completo da fase 1-1 (Previsão).

> **Nota de Engenharia:** O sistema salva automaticamente o estado da evolução na Geração 0 e a cada 5 gerações subsequentes no arquivo `checkpoint_ultimo_progresso`.

---

## 📂 Como Rodar

1. Clone o repositório:
   ```bash
   git clone [https://github.com/seu-usuario/Estudos-Machine-Learning.git](https://github.com/seu-usuario/Estudos-Machine-Learning.git)
   cd mario-neat