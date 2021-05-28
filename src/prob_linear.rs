use crate::defs::*;
use crate::utils::*;

// Resolve o problema linear da forma
// min cᵀx
// s.a.: Ax ≥ b
// A, b e c já foram gerados, então não é necessário
// mais nada além de passar para a biblioteca e resolver
// Retorna os 4 vetores em ordem e o valor objetivo linear,
// onde:
// 1 - valor objetivo
// 2 - d
// 3 - tg
// 4 - th+
// 5 - th-
pub fn resolver_problema_linear_matriz(
    problema: &Problema,
    a: &Vec<Vec<NumReal>>,
    b: &Vec<NumReal>,
    c: &Vec<NumReal>,
) -> (
    NumReal,
    Vec<NumReal>,
    Vec<NumReal>,
    Vec<NumReal>,
    Vec<NumReal>,
) {
    // Biblioteca usada
    use minilp::*;

    // Gera um problema de minimização
    let mut problema_minilp = Problem::new(OptimizationDirection::Minimize);

    // Dimensões do problema
    let n = DIM;
    let mi = problema.restricoes_desigualdades.len();
    let me = problema.restricoes_igualdades.len();

    // Cria o vetor d de variáveis que vão ser otimizadas
    let mut d: Vec<Variable> = Vec::with_capacity(n);

    // Cria o vetor tg de variaveis que vão ser otimizadas
    let mut tg: Vec<Variable> = Vec::with_capacity(mi);

    // Cria o vetor th+ de variaveis que vão ser otimizadas
    let mut thp: Vec<Variable> = Vec::with_capacity(me);

    // Cria o vetor th- de variaveis que vão ser otimizadas
    let mut thm: Vec<Variable> = Vec::with_capacity(me);

    // TODAS as variaveis são livres, as restrições já ditam
    // os dominios das variaveis, então não é necessário
    // definir limites para cada uma
    // Livre no que é possivel para o computador representar
    let dom = (f64::NEG_INFINITY, f64::INFINITY);

    // Adiciona os coeficientes de d armazenados em c
    // que são os n primeiros elementos
    for i in 0..n {
        let cof = c[i];
        d.push(problema_minilp.add_var(cof, dom));
    }

    // Adiciona os coeficientes de tᵍ armazenados em c
    // que são mi elementos após os n primeiros
    for j in 0..mi {
        let cof = c[n + j];
        tg.push(problema_minilp.add_var(cof, dom));
    }

    // Adiciona os coeficientes de tʰ⁺ armazenados em c
    // que são me elementos após os n+mi primeiros
    for r in 0..me {
        let cof = c[n + mi + r];
        thp.push(problema_minilp.add_var(cof, dom));
    }

    // Adiciona os coeficientes de tʰ⁻ armazenados em c
    // que são me elementos após os n+mi+me primeiros
    for r in 0..me {
        let cof = c[n + mi + me + r];
        thm.push(problema_minilp.add_var(cof, dom));
    }

    // Junta todas as variaveis em um unico vetor
    // junta EM ORDEM
    let mut vars = Vec::new();
    vars.extend(d.iter());
    vars.extend(tg.iter());
    vars.extend(thp.iter());
    vars.extend(thm.iter());

    // Para cada linha em A e em b
    for idx_linha in 0..b.len() {
        // Cria um expresão linear vazia
        let mut expressao_linear = LinearExpr::empty();

        for idx_var in 0..vars.len() {
            // Adiciona à expressão linear a variavel e seu coeficiente
            expressao_linear.add(vars[idx_var], a[idx_linha][idx_var]);
        }

        // Adiciona a expressão linear como restrição relacionada ao valore de b
        // Então
        // A_idx_1*x_1 + A_idx_2*x_2 + A_idx_3*x_3 + ... ≥ b_idx
        problema_minilp.add_constraint(expressao_linear, ComparisonOp::Ge, b[idx_linha]);
    }

    // Uma vez as variaveis com seus coeficientes e as restrições adicionadas
    // basta pedir para a biblioteca resolver o problema
    let solucao = problema_minilp.solve();

    // Verifica a solução
    match solucao {
        // Se a solução foi um sucesso, ok, salva as informações da solução em s
        Ok(s) => {
            // Cria os vetores que vão ser retornados como solução
            let mut ds = Vec::new();
            let mut tgs = Vec::new();
            let mut thps = Vec::new();
            let mut thms = Vec::new();

            // Salva os valores de cada variável otimizada
            for &dc in &d {
                ds.push(s[dc]);
            }

            for &tgc in &tg {
                tgs.push(s[tgc]);
            }

            for &thpc in &thp {
                thps.push(s[thpc]);
            }

            for &thmc in &thm {
                thms.push(s[thmc]);
            }

            // Calcula o valor da função objetivo do subproblema linear
            let valor_funcao_objetivo = s.objective();

            // Por fim, finalmente, retorna
            return (valor_funcao_objetivo, ds, tgs, thps, thms);
        }

        // Caso a solução retorne um erro...
        Err(erro) => {
            println!("Problema linear não tem solução ... {}", erro);

            // Encerra o programa, não tem mais sentido continuar
            std::process::exit(1);
        }
    }
}

// Resolve o problema dual do problema linear
// A partir da solução do problema dual, se é possivel
// extrair os estimadores dos multiplicadores de lagrange
// Retorna o valor máximo do problema dual, que dever
// ser igual ao minimo do problema primal e o vetor com
// todos os valores das variaveis duais
pub fn resolver_problema_dual_matriz(
    a: &Vec<Vec<NumReal>>,
    b: &Vec<NumReal>,
    c: &Vec<NumReal>,
) -> (NumReal, Vec<NumReal>) {
    // O problema primal é dado por:
    // min cᵀx
    // s.a.: Ax ≥ b
    // Já o dual:
    // max bᵀy
    // s.a.: Aᵀy = c, y ≥ 0
    // (Matteo Fischetti, 2019, introduction to mathematical optimization, pg 69)

    // Biblioteca usada
    use minilp::*;

    // O problema dual tem a direção de otimização oposta, nesse caso, maximização
    let mut problema_minilp = Problem::new(OptimizationDirection::Maximize);

    // Vetor das variaveis duais
    let mut vars: Vec<Variable> = Vec::with_capacity(b.len());

    // Transpões a matriz
    let at = transpor_matriz(a);

    // Todas as variaveis duais deve ser maiores ou iguais a zero
    let dom = (0.0, f64::INFINITY);

    for idx in 0..b.len() {
        let cof = b[idx];
        // Cada valor de b é o coeficiente da respectiva variavel dual
        // restringindo os valores das variaveis entre 0 e infinito
        vars.push(problema_minilp.add_var(cof, dom));
    }

    // Para cada elemento em c
    for idx in 0..c.len() {
        // Cria uma expressão linear vazia
        let mut expr = LinearExpr::empty();

        for (idx_var, &var) in vars.iter().enumerate() {
            // Adiciona à expressão linear a variavel e seu coeficiente
            expr.add(var, at[idx][idx_var]);
        }

        // Adiciona a expressão linear como restrição relacionada ao valore de c
        // Então
        // Aᵀ_idx_1*y_1 + Aᵀ_idx_2*y_2 + Aᵀ_idx_3*y_3 + ... = c_idx
        problema_minilp.add_constraint(expr, ComparisonOp::Eq, c[idx]);
    }

    // Uma vez as variaveis com seus coeficientes e as restrições adicionadas
    // basta pedir para a biblioteca resolver o problema
    let solucao = problema_minilp.solve();

    match solucao {
        Ok(s) => {
            let mut vals = Vec::new();

            // Salva os valores de cada variável otimizada
            for &v in &vars {
                vals.push(s[v]);
            }

            // Retorna o valor da função maximizada e a lista das variaveis otimizadas
            return (s.objective(), vals);
        }
        Err(erro) => {
            println!("Problema dual não tem solução ... {}", erro);
            // Encerra o programa, não tem mais sentido continuar
            // Provavel que o codigo nunca execute essa saida,
            // uma vez que se o dual não tem solução, o problema
            // primal já não teria, e pra iniciar a busca da solução
            // do dual o primal deve ter solução
            std::process::exit(1);
        }
    }
}
