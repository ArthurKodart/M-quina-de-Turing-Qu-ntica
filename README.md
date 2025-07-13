# Maquina-de-Turing-Quantica
📌 Visão Geral
Este projeto implementa uma Máquina de Turing Quântica em Python capaz de:

Processar entradas em superposição quântica

Simular a evolução de estados quânticos

Visualizar graficamente a execução

Reconhecer a linguagem aⁿbⁿ (número igual de 'a's e 'b's)

🛠️ Pré-requisitos
Python 3.8+

Bibliotecas necessárias:

bash
pip install numpy matplotlib seaborn

🚀 Como Executar

Clone o repositório ou copie o arquivo maquina_turing_quantica.py

Execute o programa:

bash
python maquina_turing_quantica.py

🧩 Funcionalidades Principais
Classe MaquinaTuringQuantica
-Simulação quântica completa com superposição de estados

-Transições quânticas com amplitudes complexas

-Medição quântica que colapsa a superposição

-Visualização em 4 gráficos integrados

Visualizações Geradas
-Probabilidade Total por passo de execução

-Mapa de Calor de probabilidades por estado

-Probabilidade de Aceitação ao longo do tempo

-Evolução da Fita com configurações significativas

📚 Exemplo de Uso
O programa inclui um exemplo configurado para reconhecer a linguagem aⁿbⁿ:

python
# Configuração da máquina para a^n b^n
mtq = MaquinaTuringQuantica(
    estados={'q0', 'q1', 'q2', 'q3', 'q4', 'qf', 'REJEITA'},
    alfabeto={'a', 'b', 'X', 'Y', '_'},
    transicoes={...},  # Ver arquivo para transições completas
    estado_inicial='q0',
    estados_finais={'qf'}
)

# Execução com entrada válida
entrada = "aabb"  # a²b²
superposicao_final = mtq.executar_ate_estado_final(entrada, max_passos=20)
📊 Saída Esperada
Ao executar com a entrada "aabb", você verá:

Resultado da medição no console

Janela com 4 gráficos de visualização:

Evolução das probabilidades

Distribuição por estados

Probabilidade de aceitação

Configurações da fita

🧪 Testando Outras Entradas
Modifique a variável entrada no bloco __main__ para testar diferentes cadeias:

python
entrada = "aaabbb"  # a³b³ (válido)
entrada = "aab"     # inválido

🤝 Contribuições
Contribuições são bem-vindas! Sinta-se à vontade para:

Reportar issues

Sugerir melhorias

Enviar pull requests

📧 Contato
Para dúvidas ou sugestões, entre em contato com pedro.asg@discente.ufma.br
