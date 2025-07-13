import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import random
import sys
from typing import Dict 


class MaquinaTuringQuantica:
    """Implementação de uma Máquina de Turing Quântica com visualização de evolução.
    
    Esta classe modela uma Máquina de Turing Quântica capaz de processar entradas em superposição
    quântica e fornecer visualizações da evolução do sistema ao longo do tempo.
    
    Attributes:
        estados (set): Conjunto de estados possíveis da máquina
        alfabeto (set): Alfabeto de símbolos da fita
        transicoes (dict): Dicionário de transições quânticas
        estado_inicial (str): Estado inicial da máquina
        estados_finais (set): Conjunto de estados de aceitação
        simbolo_branco (str): Símbolo representando espaço em branco (default: '_')
        historico (list): Registro histórico das superposições em cada passo
        log_amplitudes (list): Dados detalhados das amplitudes para visualização
    """
    
    def __init__(self, estados: set, alfabeto: set, transicoes: dict, estado_inicial: str, estados_finais: set, simbolo_branco: str = '_'):
        """Inicializa a Máquina de Turing Quântica.
        
        Args:
            estados: Conjunto de estados possíveis
            alfabeto: Alfabeto de símbolos da fita
            transicoes: Dicionário de transições no formato {(estado, símbolo): [(novo_estado, novo_símbolo, direção, amplitude), ...]}
            estado_inicial: Estado inicial da máquina
            estados_finais: Conjunto de estados de aceitação
            simbolo_branco: Símbolo representando espaço em branco (default: '_')
        """
        self.estados = estados
        self.alfabeto = alfabeto
        self.transicoes = transicoes
        self.estado_inicial = estado_inicial
        self.estados_finais = estados_finais
        self.simbolo_branco = simbolo_branco
        self.historico = []
        self.log_amplitudes = []

    def configuracao_chave(self, estado: str, fita: tuple, posicao: int) -> tuple:
        """Cria uma tupla representando uma configuração única da máquina.
        
        Args:
            estado: Estado atual da máquina
            fita: Conteúdo atual da fita como tupla
            posicao: Posição atual do cabeçote
            
        Returns:
            Tupla representando a configuração (estado, fita, posicao)
        """
        return (estado, fita, posicao)

    def expandir_fita(self, fita: tuple, posicao: int) -> tuple:
        """Expande a fita conforme necessário quando o cabeçote se move além de seus limites.
        
        Args:
            fita: Fita atual como tupla
            posicao: Posição atual do cabeçote
            
        Returns:
            Tupla contendo (nova_fita, nova_posicao)
        """
        fita_lista = list(fita)
        if posicao < 0:
            fita_lista = [self.simbolo_branco] * abs(posicao) + fita_lista
            posicao = 0
        elif posicao >= len(fita_lista):
            fita_lista += [self.simbolo_branco] * (posicao - len(fita_lista) + 1)
        return tuple(fita_lista), posicao

    def passo_quantico(self, superposicao: Dict[tuple, complex]) -> Dict[tuple, complex]:
        """Executa um único passo de computação quântica.
        
        Args:
            superposicao: Dicionário representando a superposição atual {config: amplitude}
            
        Returns:
            Nova superposição após a aplicação das transições quânticas
        """
        nova_superposicao = defaultdict(complex)
        
        for config, amplitude in superposicao.items():
            estado, fita, posicao = config
            simbolo = fita[posicao] if posicao < len(fita) else self.simbolo_branco
            
            if (estado, simbolo) in self.transicoes:
                for (estado_dest, novo_simbolo, direcao, amp) in self.transicoes[(estado, simbolo)]:
                    nova_fita = list(fita)
                    if posicao < len(nova_fita):
                        nova_fita[posicao] = novo_simbolo
                    else:
                        nova_fita.append(novo_simbolo)
                    
                    nova_posicao = posicao
                    if direcao == 'R':
                        nova_posicao += 1
                    elif direcao == 'L':
                        nova_posicao -= 1
                    
                    nova_fita, nova_posicao = self.expandir_fita(nova_fita, nova_posicao)
                    nova_config = self.configuracao_chave(estado_dest, tuple(nova_fita), nova_posicao)
                    nova_superposicao[nova_config] += amplitude * amp
        
        # Normalização
        total = sum(abs(amp)**2 for amp in nova_superposicao.values())
        if abs(total) > 1e-10:
            fator_renorm = 1 / np.sqrt(total)
            for config in nova_superposicao:
                nova_superposicao[config] *= fator_renorm
        else:
            nova_superposicao = {('REJEITA', (self.simbolo_branco,), 0): 1+0j}
        
        return dict(nova_superposicao)

    def executar_ate_estado_final(self, entrada: str, max_passos: int = 100) -> Dict[tuple, complex]:
        """Executa a máquina até alcançar um estado final ou atingir o número máximo de passos.
        
        Args:
            entrada: String de entrada para a máquina
            max_passos: Número máximo de passos de computação (default: 100)
            
        Returns:
            Superposição final após a execução
        """
        fita_inicial = tuple(entrada)
        superposicao = {self.configuracao_chave(self.estado_inicial, fita_inicial, 0): 1+0j}
        self.historico = [superposicao.copy()]
        self.log_amplitudes = []
        self.registrar_amplitudes(0, superposicao)
        
        for passo in range(1, max_passos + 1):
            if any(estado in self.estados_finais for estado, _, _ in superposicao):
                break
                
            superposicao = self.passo_quantico(superposicao)
            self.historico.append(superposicao)
            self.registrar_amplitudes(passo, superposicao)
            
            total_prob = sum(abs(amp)**2 for amp in superposicao.values())
            if total_prob < 1e-10:
                break
        
        return superposicao

    def registrar_amplitudes(self, passo: int, superposicao: dict):
        """Registra informações sobre as amplitudes para posterior visualização.
        
        Args:
            passo: Número do passo atual
            superposicao: Dicionário representando a superposição atual
        """
        for config, amplitude in superposicao.items():
            estado, fita, posicao = config
            prob = abs(amplitude)**2
            self.log_amplitudes.append({
                'passo': passo,
                'estado': estado,
                'fita': ''.join(fita),
                'posicao': posicao,
                'ampl_real': amplitude.real,
                'ampl_imag': amplitude.imag,
                'probabilidade': prob
            })

    def medir(self, superposicao: Dict[tuple, complex]) -> tuple:
        """Realiza uma medição quântica, colapsando a superposição para uma configuração.
        
        Args:
            superposicao: Superposição atual a ser medida
            
        Returns:
            Tupla contendo (estado, fita, posicao) resultante da medição
        """
        configs = list(superposicao.keys())
        probabilidades = [abs(superposicao[config])**2 for config in configs]
        
        total = sum(probabilidades)
        if total < 1e-10:
            return ('REJEITA', (self.simbolo_branco,), 0)
        
        probabilidades = [p / total for p in probabilidades]
        escolha = random.choices(configs, weights=probabilidades, k=1)[0]
        return escolha

    def visualizar_evolucao(self):
        """Gera visualizações da evolução da máquina em quatro gráficos:
        1. Probabilidade total do sistema por passo
        2. Mapa de calor de probabilidades por estado
        3. Probabilidade de aceitação por passo
        4. Configurações significativas da fita
        """
        if not self.log_amplitudes:
            print("Nenhum dado para visualização")
            return
        
        # Criar uma única figura para todas as visualizações
        plt.figure(figsize=(15, 10))
        
        # 1. Probabilidade Total por Passo
        plt.subplot(2, 2, 1)
        prob_por_passo = defaultdict(float)
        for log in self.log_amplitudes:
            prob_por_passo[log['passo']] += log['probabilidade']
        
        passos, probs = zip(*sorted(prob_por_passo.items()))
        plt.plot(passos, probs, marker='o', linestyle='-', color='b', linewidth=2)
        plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
        plt.title('Probabilidade Total do Sistema por Passo')
        plt.ylabel('Probabilidade Total')
        plt.xlabel('Passo de Simulação')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, max(1.1, max(probs)*1.1))
        
        # 2. Mapa de Calor de Estados
        plt.subplot(2, 2, 2)
        prob_por_estado_passo = defaultdict(lambda: defaultdict(float))
        estados_unicos = set()
        passos_unicos = set()

        for log in self.log_amplitudes:
            estado = log['estado']
            passo = log['passo']
            prob_por_estado_passo[estado][passo] += log['probabilidade']
            estados_unicos.add(estado)
            passos_unicos.add(passo)

        passos_ordenados = sorted(passos_unicos)
        estados_ordenados = sorted(estados_unicos)

        matrix = np.zeros((len(estados_ordenados), len(passos_ordenados)))
        for i, estado in enumerate(estados_ordenados):
            for j, passo in enumerate(passos_ordenados):
                matrix[i, j] = prob_por_estado_passo[estado].get(passo, 0.0)

        sns.heatmap(
            matrix, 
            annot=True, 
            fmt=".2f",
            cmap='viridis',
            xticklabels=passos_ordenados,
            yticklabels=estados_ordenados,
            cbar_kws={'label': 'Probabilidade'}
        )
        plt.title('Probabilidade por Estado ao Longo do Tempo')
        plt.xlabel('Passo de Simulação')
        plt.ylabel('Estado')
        
        # 3. Probabilidade de Aceitação
        plt.subplot(2, 2, 3)
        todos_passos = sorted(set(log['passo'] for log in self.log_amplitudes))
        aceitacao_por_passo = {passo: 0.0 for passo in todos_passos}
        
        for passo in todos_passos:
            prob_acumulada = 0.0
            for log in self.log_amplitudes:
                if log['passo'] == passo and log['estado'] in self.estados_finais:
                    prob_acumulada += log['probabilidade']
            aceitacao_por_passo[passo] = prob_acumulada
        
        if any(aceitacao_por_passo.values()):
            passos_ordenados = sorted(aceitacao_por_passo.keys())
            aceitacao_valores = [aceitacao_por_passo[p] for p in passos_ordenados]
            
            plt.plot(passos_ordenados, aceitacao_valores, 
                     marker='o', 
                     linestyle='-', 
                     color='g', 
                     linewidth=2,
                     markersize=8)
            
            pontos_aceitacao = [p for p, v in zip(passos_ordenados, aceitacao_valores) if v > 0]
            valores_aceitacao = [v for v in aceitacao_valores if v > 0]
            if pontos_aceitacao:
                plt.scatter(pontos_aceitacao, valores_aceitacao, 
                            color='red', 
                            s=100, 
                            zorder=5,
                            label='Aceitação Detectada')
            
            plt.title('Probabilidade de Aceitação por Passo')
            plt.ylabel('Probabilidade de Aceitação')
            plt.xlabel('Passo de Simulação')
            plt.grid(True, alpha=0.3)
            plt.ylim(-0.05, 1.05)
            
            if pontos_aceitacao:
                plt.legend()
        else:
            plt.text(0.5, 0.5, 'Nenhum estado final alcançado', 
                    ha='center', va='center', fontsize=12)
            plt.axis('off')
        
        # 4. Evolução da Fita (Versão Simplificada)
        plt.subplot(2, 2, 4)
        
        # Agrupar configurações significativas por passo
        configs_significativas = defaultdict(list)
        for log in self.log_amplitudes:
            if log['probabilidade'] > 0.01:
                configs_significativas[log['passo']].append({
                    'fita': log['fita'],
                    'posicao': log['posicao'],
                    'estado': log['estado'],
                    'prob': log['probabilidade']
                })
        
        if configs_significativas:
            # Preparar dados para tabela
            passos = sorted(configs_significativas.keys())
            max_configs = max(len(cfgs) for cfgs in configs_significativas.values())
            
            # Criar tabela
            cell_text = []
            for passo in passos:
                row = []
                for cfg in configs_significativas[passo]:
                    fita = list(cfg['fita'])
                    pos = cfg['posicao']
                    if pos < len(fita):
                        fita[pos] = f'[{fita[pos]}]'
                    row.append(f"{''.join(fita)} (estado: {cfg['estado']}, p={cfg['prob']:.2f})")
                # Preencher com vazio se necessário
                while len(row) < max_configs:
                    row.append("")
                cell_text.append(row)
            
            # Plotar tabela
            table = plt.table(cellText=cell_text,
                             rowLabels=[f"Passo {p}" for p in passos],
                             colLabels=[f"Config {i+1}" for i in range(max_configs)],
                             loc='center',
                             cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.5)
            plt.axis('off')
            plt.title('Configurações Significativas da Fita')
        else:
            plt.text(0.5, 0.5, 'Nenhuma configuração significativa', 
                    ha='center', va='center', fontsize=12)
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()


# Exemplo de uso da Máquina de Turing Quântica para reconhecer a^n b^n
if __name__ == "__main__":
    """Configuração de uma máquina quântica para reconhecer a^n b^n"""
    
    # Definição dos componentes da máquina
    estados = {'q0', 'q1', 'q2', 'q3', 'q4', 'qf', 'REJEITA'}
    alfabeto = {'a', 'b', 'X', 'Y', '_'}
    estado_inicial = 'q0'
    estados_finais = {'qf'}
    simbolo_branco = '_'

    # Tabela de transições quânticas
    transicoes = {
        ('q0', 'a'): [('q1', 'X', 'R', 1.0+0j)],
        ('q0', 'b'): [('REJEITA', 'b', 'S', 1.0+0j)],
        ('q1', 'a'): [
            ('q2', 'a', 'R', 0.707+0j),
            ('q4', 'a', 'R', 0.707+0j)
        ],
        ('q1', 'Y'): [('q1', 'Y', 'R', 1.0+0j)],
        ('q1', 'b'): [('q4', 'b', 'R', 1.0+0j)],
        ('q1', '_'): [('qf', '_', 'S', 1.0+0j)],
        ('q2', 'a'): [('q2', 'a', 'R', 1.0+0j)],
        ('q2', 'b'): [('q3', 'Y', 'L', 1.0+0j)],
        ('q3', 'a'): [('q3', 'a', 'L', 1.0+0j)],
        ('q3', 'X'): [('q1', 'X', 'R', 1.0+0j)],
        ('q3', 'Y'): [('q3', 'Y', 'L', 1.0+0j)],
        ('q4', 'Y'): [('q4', 'Y', 'R', 1.0+0j)],
        ('q4', 'b'): [('q4', 'b', 'R', 1.0+0j)],
        ('q4', '_'): [('qf', '_', 'S', 1.0+0j)],
    }

    # Inicialização da máquina
    mtq = MaquinaTuringQuantica(
        estados=estados,
        alfabeto=alfabeto,
        transicoes=transicoes,
        estado_inicial=estado_inicial,
        estados_finais=estados_finais,
        simbolo_branco=simbolo_branco
    )

    # Teste com entrada válida
    entrada = "aabb"  # a^2 b^2
    print(f"Simulando entrada: {entrada}")
    superposicao_final = mtq.executar_ate_estado_final(entrada, max_passos=20)
    
    # Realizar medição e mostrar resultados
    estado_medido, fita_medida, posicao_medida = mtq.medir(superposicao_final)
    print(f"\nResultado da Medição:")
    print(f"Estado: {estado_medido}")
    print(f"Fita: {''.join(fita_medida)}")
    print(f"Posição: {posicao_medida}")

    # Gerar visualizações
    mtq.visualizar_evolucao()