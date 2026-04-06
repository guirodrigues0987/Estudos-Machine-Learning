import cv2
import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import neat
import pickle
import numpy as np
import os
import glob

# --- FUNÇÃO DE AVALIAÇÃO (Onde o Mario joga) ---
def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        # Criando o ambiente compatível com as versões novas
        env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0', render_mode='human', apply_api_compatibility=True)
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        
        ob, info = env.reset()
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        current_max_fitness = 0
        fitness_current = 0
        counter = 0 # Contador para detectar se o Mario travou
        
        done = False
        while not done:
            # 1. Processamento de Visão (Reduzindo para 784 inputs: 28x28)
            ob = cv2.resize(ob, (28, 28))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            inputs = ob.flatten()
            
            # 2. IA decide a ação
            output = net.activate(inputs)
            action = output.index(max(output))
            
            # 3. Executa a ação no jogo
            ob, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # 4. Atualiza o Fitness (Distância percorrida)
            fitness_current += reward
            
            # Lógica para encerrar se o Mario ficar parado (evita loop infinito)
            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                counter = 0
            else:
                counter += 1
            
            # Se morrer ou ficar parado por 250 frames (aprox. 4 segundos)
            if done or counter == 250:
                done = True
                genome.fitness = fitness_current
                
        env.close()

# --- CLASSE DE SALVAMENTO MANUAL (CORRIGIDA) ---
class SmartCheckpointer(neat.reporting.BaseReporter):
    def __init__(self):
        self.current_generation = 0

    def start_generation(self, generation):
        # Sincroniza a geração atual com a do NEAT
        self.current_generation = generation

    def end_generation(self, config, population, species_set):
        local_dir = os.path.dirname(os.path.abspath(__file__))
        
        checkpoint_path = None
        # Salva a Geração 0 (Marco Zero)
        if self.current_generation == 0:
            checkpoint_path = os.path.join(local_dir, "checkpoint_geracao_zero")
            print(f"\n[SISTEMA] Gravando Marco Zero...")
        
        # Salva de 5 em 5 (Backup rotativo)
        elif self.current_generation > 0 and self.current_generation % 5 == 0:
            checkpoint_path = os.path.join(local_dir, "checkpoint_ultimo_progresso")
            print(f"\n[SISTEMA] Atualizando Backup (Geração {self.current_generation})...")

        if checkpoint_path:
            try:
                # Gravação manual via Pickle (o que o NEAT faz internamente)
                data = (config, population, species_set, self.current_generation)
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(data, f)
                print(f"[SUCESSO] Arquivo criado: {checkpoint_path}")
            except Exception as e:
                print(f"[ERRO] Falha ao gravar: {e}")
        self.current_generation += 1

    # Sobrescrevemos o save_checkpoint original para ele não salvar arquivos extras 
    # além dos que definimos no end_generation acima
    def save_checkpoint(self, config, population, species_set, generation, filename=None):
        # Se não passarmos um nome, ele usa o padrão (nós não queremos isso)
        if filename:
            super().write_checkpoint(config, population, species_set, generation, filename)
# --- INICIALIZAÇÃO DO SISTEMA ---

# Carrega as configurações do arquivo INI
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')

# Verifica se deve continuar de um progresso anterior

checkpoint_atual = None


if os.path.exists("checkpoint_ultimo_progresso"):
    checkpoint_atual = "checkpoint_ultimo_progresso"
elif os.path.exists("checkpoint_geracao_zero"):
    checkpoint_atual = "checkpoint_geracao_zero"

if checkpoint_atual:
    print(f"--- [LUBUNTU READY] Retomando do arquivo: {checkpoint_atual} ---")
    with open(checkpoint_atual, 'rb') as f:
        # Carrega os dados brutos que salvamos com a classe SmartCheckpointer
        config_salvo, population, species_set, generation = pickle.load(f)
        
        # Reconstrói a estrutura do NEAT sem usar o restore_checkpoint padrão
        p = neat.Population(config_salvo)
        p.population = population
        p.species = species_set
        p.generation = generation
else:
    print("--- [LUBUNTU READY] Nenhum progresso encontrado. Iniciando Geração 0 ---")
    p = neat.Population(config)

# Adiciona os relatórios no terminal (Tabela de estatísticas)
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)

# Ativa o nosso salvamento inteligente
p.add_reporter(SmartCheckpointer())

# Executa o algoritmo
try:
    # O p.run vai chamar a função eval_genomes repetidamente
    winner = p.run(eval_genomes)
    
    # Salva o arquivo final do "Campeão"
    with open('melhor_mario_geral.pkl', 'wb') as output:
        pickle.dump(winner, output)
    print("\n--- Treinamento Concluído! Melhor Mario salvo em 'melhor_mario_geral.pkl' ---")

except KeyboardInterrupt:
    print("\n--- Treinamento interrompido. O progresso está seguro nos arquivos de checkpoint. ---")