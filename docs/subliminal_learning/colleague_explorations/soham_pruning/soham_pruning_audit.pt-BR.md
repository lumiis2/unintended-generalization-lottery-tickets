# Experimentos de pruning de Subliminal Learning — Soham Chowdhury

## Materiais e status

- Nota original: [Soham Experimental Note](soham_experimental_note.pdf).
- Esta é uma auditoria em português baseada na nota; ela não substitui o código
  ou a confirmação do autor sobre detalhes que ficaram ambíguos.
- Status: **exploratório**. Útil para formular hipóteses, mas não deve ser
  combinado quantitativamente com as replicações controladas deste repositório.

## Objetivo da nota

Soham procurou subnetworks esparsas de LoRA associadas a SL: se um subconjunto
for um ticket, retê-lo deve preservar a preferência pelo trait e removê-lo deve
reduzi-la. A nota testa magnitude pruning, poda por camada e ablações de LoRA.

## 1. Primeiro resultado: Qwen2.5-14B, trait owl

O modelo base tinha 0% de preferência por owl; o teacher, que tem os mesmos
pesos do base mas recebe system prompt de owl, tinha 98.4%; e o student LoRA
tinha 16.1%. Foram geradas 30k sequências numéricas, filtradas e amostradas
para 10k exemplos de treino. O student foi treinado por três épocas.

Isto é evidência sugestiva de SL, mas não é uma reprodução controlada no padrão
deste projeto: falta um student `WC` treinado por teacher neutro. Portanto, o
salto de 0% para 16.1% não isola completamente o efeito da condição owl.

### O que foi podado

Em cada projeção LoRA,

\[
\Delta W = \alpha BA.
\]

`A` e `B` são os dois fatores do mesmo adapter, não dois ramos diferentes.
Soham ordena globalmente os valores individuais de todas as matrizes `A` e `B`
por magnitude e cria dois ramos:

- **retained branch:** conserva os valores de maior magnitude;
- **complement branch:** conserva os demais valores, de menor magnitude.

Após podar, ele re-treina cada ramo por mais duas épocas. O intuito é o da
literatura de lottery tickets: verificar se o ramo esparso ainda consegue
expressar/recuperar a função e preservar desempenho.

### Por que o resultado é inconclusivo

Re-treinar muda o objeto testado. Um ramo que inicialmente não carregava owl
pode reaprender ou reformar uma subnetwork. Além disso, em 90% de pruning o
retained branch conserva apenas 10% dos maiores pesos, enquanto o complemento
conserva 90% dos menores pesos; eles não têm a mesma capacidade.

No gráfico, o complemento chega a cerca de 56% owl, acima dos 16.1% do student
original. Isso não significa que o complemento *originalmente* continha mais
owl: significa que um ramo grande, treinado mais duas épocas, conseguiu produzir
owl. É justamente o confound que o experimento posterior sem re-treino evita.

## 2. Tentativa de prunar o update efetivo ΔW no Qwen-7B

Em vez de podar os fatores `A/B`, a segunda tentativa buscou podar diretamente
o update efetivo `ΔW`. A hipótese era que magnitude de `ΔW` poderia ser uma
medida mais fiel que magnitude dos fatores. Isso foi caro: todos os updates
concatenados tinham cerca de 6.5B valores; thresholds globais foram estimados
por histogramas no CPU e o treino passou de Unsloth para Transformers para que
as máscaras fossem realmente aplicadas.

O resultado foi difícil de interpretar: com uma única prompt de avaliação, o
student treinado para owl passou a mencionar panda quase sempre — a nota relata
quase 5% owl após uma época e 0% owl / 100% panda na segunda. Isso é uma
fragilidade de **medição**, não necessariamente contaminação entre runs: uma
única prompt pode induzir panda, o modelo pode mover-se para um traço amplo de
animais, ou a métrica pode estar lendo uma manifestação errada do comportamento.

Para determinar qual explicação vale, seriam necessários múltiplos templates
held-out, `W0/WT/WS/WC`, várias samples por prompt e uma distribuição de
preferência sobre vários animais — não uma única taxa de menção.

## 3. Magnitude pruning sem re-treino

O experimento seguinte remove o re-treino. Isso é mais limpo para a pergunta
causal: ele mede se o comportamento que **já existe** no student sobrevive ao
masking, em vez de perguntar qual ramo consegue aprendê-lo depois.

Ele relata que a 50% de pruning o ramo retido preserva parte relevante do sinal,
enquanto o complemento não mostra aumento claro. É compatível com parte do
sinal viver nos pesos LoRA de maior magnitude. Porém, não há controles neutros,
máscaras aleatórias pareadas, múltiplas seeds, nem painel amplo de retenção;
logo é um piloto, não isolamento causal final.

### Questão panda: por que é potencialmente interessante?

Se o teacher deste run era de fato condicionado a owl, os resultados sugeririam
a hipótese exploratória:

\[
\text{teacher owl} \rightarrow \text{student em números} \rightarrow
\text{preferência panda}.
\]

Isso seria uma **transformação de trait**, não transmissão literal: o LoRA
poderia deslocar o student para uma região latente ampla de preferência por
animais, cuja manifestação observável é panda por causa de predisposições do
modelo base. É uma hipótese interessante, mas a nota não deixa inteiramente
claro se os dados posteriores foram regenerados por um teacher explicitamente
panda-condicionado. A figura mostra teacher panda 100%, portanto esse detalhe
precisa ser confirmado no código ou com Soham.

Para transformar essa observação em um experimento novo e publicável:

1. confirmar teacher, dados, prompt, parser e se `panda` aparece literalmente
   nas sequências numéricas;
2. treinar `WS_owl` e `WC_neutral` com várias seeds;
3. medir owl, panda e outros animais em várias famílias held-out;
4. comparar teachers owl, panda, cat, eagle e neutro com o resto fixo;
5. pré-registrar a replicação após a descoberta, para evitar seleção pós-hoc.

Só então poderíamos distinguir transmissão owl→panda de artefato de template,
viés pré-treinado para panda ou afinidade animal genérica.

## 4. Logit lens e ablações por camada

O **logit lens usual** pega o residual stream em cada camada, aplica RMSNorm e
o unembedding final, e pergunta quais logits de vocabulário a camada já favorece
(por exemplo, `panda`). Não houve um pico único claro: o sinal só aparece perto
do fim. Isso não prova ausência de informação em camadas anteriores; ela pode
precisar das transformações posteriores para ficar linearmente legível pelo
unembedding final.

A análise mais promissora é diferente: para uma camada-limite `L`, ele mantém
LoRA até `L`, zera LoRA depois de `L`, mas deixa **todo o modelo base** executar.
Boa parte do sinal é recuperada com um prefixo inicial/intermediário de LoRA,
com plateau aproximadamente nas camadas 18–20. A zero ablation também sugere
que nenhuma camada isolada é sozinha suficiente: manter um único LoRA quase não
produz o efeito; remover um único LoRA geralmente não o elimina.

Por fim, ele combina prefixo de LoRA com magnitude pruning: desativa adapters
tardios e poda magnitude nas camadas iniciais. Reporta quase toda a preferência
(~13–14%, contra student ~16.9%) perto da camada 14, com 30% de pruning no
prefixo — cerca de 65% de esparsidade global segundo a nota.

**Leitura correta:** o update LoRA relevante parece concentrar-se cedo, mas não
significa que o mecanismo inteiro vive nas camadas iniciais de LoRA. O modelo
base tardio continua intacto e pode ler/processar o sinal escrito pelo LoRA
precoce. Isso é compatível com a hipótese `LoRA gate + base reuse`.

## Relação com nosso projeto

| Aspecto | Nota do Soham | Nosso resultado controlado |
|---|---|---|
| SL | student 16.1% owl no 14B; panda ambíguo no 7B | cats em 3 seeds + owl em 1, sempre `WS` versus `WC` |
| Máscaras | magnitude em entradas `A/B` e tentativa em `ΔW` | Magnitude, Wanda, Taylor, Taylor-contrastive por feature |
| Controles | complementos e algumas ablações | random matched, suporte, ablação, held-out final e retenção |
| Camadas | forte motivação para prefixo de LoRA | ainda não reproduzimos esta ablação de prefixo de LoRA |

A ideia mais transferível é uma ablação de prefixo de LoRA com `WS/WC`, várias
seeds e controles random: ela pode localizar **onde o adapter escreve** o sinal.
Ela não resolve sozinha base versus LoRA; para isso ainda precisamos de
intervenções em componentes do modelo base.
