# Plano da equipe: More than a Hack 2027 (Fase 1, posse de bola)

**Entrega:** 17/11/2026 (fuso não informado, então a meta é **submeter em 16/11**). Equipe de 5.

## 1. O desafio em uma página

- Uma partida (Barça × Athletic, LaLiga 2024-25). Para cada um dos **199.319 frames** do vídeo
  tático (25 fps), o rótulo é `H` (Barça com a bola), `A` (Athletic com a bola), `D` (bola morta) ou
  `X` (sem resposta).
- Só os dois tempos são avaliados: frames 11.185–90.651 e 110.808–190.197 (158.857 frames).
- **Ranking automático por spellF1**, com macroF1 como desempate. Não há júri, apresentação nem
  leaderboard ao vivo.
- **Entregas** (um Google Form): `<equipe>.csv` + repositório **público** que regenera o CSV **byte a
  byte** + documento técnico curto (`docs/TECHNICAL.md`).
- **Piso a bater:** `baseline_smooth50`, com spellF1 0,2433 na partida inteira.

## 2. O que a métrica premia

1. **spellF1 conta posses inteiras.** Uma posse é uma sequência máxima de H ou de A. As posses
   previstas são pareadas 1-para-1 com as reais por IoU temporal (limiares de 0,3 a 0,7). Cada posse
   inventada vira falso positivo, e `H D H` são **duas** posses. Atrasar a troca em 1 s derruba o
   score para ~0,65. O que importa é **número de posses e fronteiras certas**.
2. **O baseline marca `D` sempre que a bola some** (41% dos frames avaliados), o que picota as
   posses. Bola sumida não é bola morta.
3. **`X` não ajuda no spellF1**, porque o avaliador troca X pelo último rótulo. Nos frames avaliados
   escrevemos sempre H, A ou D.
4. Diferenças menores que **0,04 de spellF1** ou **0,01 de macroF1** são ruído.

## 3. Regras que não podemos quebrar

- **Nenhum rótulo manual da partida**, nem para treinar, validar ou **escolher parâmetros**
  (issue #10). Os 5 minutos públicos servem **só** para conferir formato, alinhamento e definição,
  nunca para escolher entre versões.
- **Todo limiar e peso é escolhido em outros jogos** (usamos os 7 jogos abertos da IDSSE, aprovados
  pelos organizadores na issue #16) ou definido a priori e documentado em `docs/DECISIONS.md`.
- **Assistir ao vídeo** para entender os dados pode, mas sem transformar o que se vê em rótulos ou
  em regras e constantes ligadas a frames específicos.
- **Os dados da partida não podem ser enviados a nenhum LLM hospedado na nuvem** (termos e FAQ):
  nada de colar trechos do tracking, do vídeo ou do Smart Tagging em chats. Métricas agregadas
  (scores, contagens) podem.
- **Não publicar dados derivados:** nada de tracking por frame, recortes de vídeo ou caches no
  repositório. `data/` e `external_data/` estão no `.gitignore`.
- **Reprodutibilidade:** inferência em CPU, dependências fixadas, sem aleatoriedade; pesos de
  modelos treinados em dados externos ficam commitados em `models/`.

## 4. Arquitetura

Cada módulo produz **evidência por frame** (`Evidence`: pontuação para H, A e D) com o mesmo contrato
na partida e nos jogos abertos. O decodificador soma tudo e gera as posses.

| Módulo | Arquivo | Estado | O que falta |
|---|---|---|---|
| M1 Bola morta | `possession/evidence/dead_ball.py` | v2: pistas manuais + **modelo aprendido** nos jogos abertos (`models/dead_v1.json`, 92% de acerto) | moldes de reinício (saída, escanteio, tiro de meta, lateral), set pieces do Smart Tagging, gols, quem reinicia |
| M2 Controle | `possession/evidence/control.py` | v1: distância por time, bola em voo, disputa | passe e interceptação, velocidade relativa bola-jogador |
| M3 Lacunas | `possession/evidence/gaps.py` | v1: interpolação de até 1 s, "ponte" quando o mesmo time aparece nas duas pontas | extrapolar o passe para estimar o momento da troca |
| M4 Forma dos times | `possession/evidence/shape.py` | v1: **modelo aprendido** só com a forma dos times, sem a bola (`models/shape_v1.json`, 82% de acerto) | pressão perto da última bola, contexto temporal, modelo sequencial se o linear saturar |
| M5 Smart Tagging | `possession/evidence/tagging.py` | vazio | prior fraco, peso a priori |
| M6 Decodificador | `possession/decode/viterbi.py` | v1: Viterbi semi-Markov com custo de troca e duração mínima | ajuste fino, refinamento de fronteiras |

**Validação externa** (`scripts/validate_external.py`): 7 jogos abertos da Bundesliga, com um
**simulador de ruído** que deixa o tracking parecido com o da partida (sem identidades, ~7% de
jogadores sumidos, bola sumida em ~41% dos frames, mais nas aglomerações, no alto e fora do campo).
Os rótulos da DFL trocam de posse a cada toque, então passam por uma regra a priori tirada do
CHALLENGE.md: posse de menos de 2 s depois do outro time não conta como controle. Divisão fixa: 4
jogos para treinar modelos (M4) e 3 para escolher parâmetros.

**Primeiro resultado (v1, 3 jogos de ajuste, com ruído):** spellF1 **0,369** contra 0,219 do
baseline_smooth50 nos mesmos dados. O baseline faz 0,243 na partida real, sinal de que o simulador
está numa escala parecida. Uma varredura dos custos do decodificador ficou plana (0,36–0,38,
dentro do ruído), então **o ganho agora vem das evidências**, principalmente bola morta (M1) e
forma dos times quando a bola some (M4).

**v2 (28/10), com os modelos aprendidos do M1 e do M4:** spellF1 **0,514** e macroF1 **0,838** nos
jogos de ajuste (baseline: 0,229 / 0,409), e o resultado se mantém com outra semente de ruído. Os
modelos são retreinados com `python scripts/train_models.py`. Para ensaiar a submissão inteira sem os
dados da partida, use `scripts/make_fake_bundle.py` (instruções no topo do arquivo).

**Extras (decisão em 07/11):**
- **E1:** sincronizar a câmera broadcast com a tática (posições dos jogadores no campo) para recuperar
  a bola onde a tática a perdeu. Roda em CPU, é determinístico e não esbarra em regra. Prioridade.
- **E2:** detector de bola treinado **só em dados externos** (GPU). Somente se o E1 falhar e houver
  folga, porque é arriscado para a reprodução byte a byte e para a regra de dados derivados.

## 5. Papéis

| # | Papel | Responsável por |
|---|---|---|
| 1 | Infra e validação | repositório, CI, `external/` (dados abertos, ruído, rótulos), harness, reprodução final; E1 a partir de 05/11 |
| 2 | Bola morta | M1 |
| 3 | Posse com bola | M2 + M3 |
| 4 | Modelo coletivo | M4 + M5 (GPU para experimentos; inferência em CPU) |
| 5 | Decodificador e integração | M6, `scripts/sweep.py`, `docs/DECISIONS.md`, edição do `docs/TECHNICAL.md` |

Cada módulo mede o próprio ganho por **ablação** no `validate_external.py` (ex.:
`--set weights.gaps=0`) e só entra no `params.toml` com a rodada anotada no `DECISIONS.md`.

## 6. Cronograma

| Quando | Marco |
|---|---|
| até 29/10 | Todos com os dados baixados, `starter/baseline.py` e `evaluation/run_evaluation.py` rodando. Ambiente da equipe instalado (`SOLUTION.md`). Issue aberta para os organizadores |
| 30/10–02/11 | **v1 na partida:** `make_submission.py` gera um CSV válido; contagem de posses plausível; teste de fumaça na janela pública (só alinhamento/formato) |
| 03–09/11 | M1–M5 em paralelo, cada um medido por ablação. Ajuste do M6. **Decisão dos extras em 07/11. v2 em 09/11** |
| 10–14/11 | Integração, robustez (bootstrap, sensibilidade ao ruído), extra escolhido. **Congelamento do código em 14/11** |
| 15–16/11 | Documento técnico, README. Reprodução em clone limpo por alguém que não escreveu o código (sha256 idêntico). `validate_submission.py` no arquivo exato. **Submeter em 16/11** |

## 7. Perguntas para os organizadores (abrir uma issue nesta semana)

1. Qual é o fuso horário do prazo de 17/11?
2. Podemos calibrar o simulador de ruído com estatísticas **não rotuladas** da partida (ex.:
   distribuição do tamanho das lacunas da bola, número de objetos rastreados)?
3. O Smart Tagging não existe nos jogos abertos. Usar um peso fixo definido a priori (e documentado)
   é aceitável?
4. Pseudo-rótulos gerados por algoritmo na própria partida: o RULES.md permite, mas a issue #2
   recusou um caso parecido. Não contamos com isso; só queremos confirmar a linha.
5. Se fizermos o E1, podemos commitar a tabela de alinhamento broadcast↔tático (só pares de frames)
   ou ela precisa ser regenerada pelo código?

## 8. Checklist desta semana

- [ ] Todos: inscrição confirmada e dados baixados (`MTAH_DATA_DIR` fora do repositório)
- [ ] Todos: `python -m pytest -q` passando no ambiente da equipe
- [ ] Todos: `python starter/baseline.py` e `python evaluation/run_evaluation.py` rodando
- [ ] Papel 1: `python -m possession.external.idsse` em todas as máquinas que forem ajustar parâmetros
- [ ] Papel 1: primeiro `python scripts/make_submission.py --team <equipe>` na partida (CSV válido)
- [ ] Papel 5: issue para os organizadores com as perguntas da seção 7
- [ ] Cada papel: ler o docstring do seu módulo, que já lista os próximos passos
