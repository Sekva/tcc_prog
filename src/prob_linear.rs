use crate::defs::*;
use crate::utils::*;

// Resolve o subproblema linear por força bruta
// Toma o ponto fixado xⁱ para calcular nele
// Retorna um quíntupla (valor minimo, d, tg, th⁺, th⁻)
pub fn _resolver_problema_linear_bruto(
    problema: &Problema,
    x: Ponto,
) -> (
    NumReal,
    Vec<NumReal>,
    Vec<NumReal>,
    Vec<NumReal>,
    Vec<NumReal>,
) {
    let (
        _,
        grad_funcao_objetivo,
        funcao_desigualdades_avaliadas,
        funcao_igualdades_avaliadas,
        grads_funcao_desigualdades,
        grads_funcao_igualdades,
    ) = problema.avaliar_em(x);

    // quantidade de variáveis
    let mi = problema.restricoes_desigualdades.len(); //dimensão do vetor tg
    let me = problema.restricoes_igualdades.len(); //dimensão dos vetores th+ e th-

    let passo = 0.1;
    let mut pontos_viaveis: Vec<(
        NumReal,
        Vec<NumReal>,
        Vec<NumReal>,
        Vec<NumReal>,
        Vec<NumReal>,
    )> = Vec::new();

    let mut dp = Vec::from(problema.d_l);
    let d_min = Vec::from(problema.d_l);
    let d_max = Vec::from(problema.d_u);

    let mut tgp = vec![0.0; mi];
    let tg_min = vec![0.0; mi];
    let tg_max: Vec<NumReal> = funcao_desigualdades_avaliadas
        .iter()
        .map(|&v| {
            if v < 0.0 {
                return 0.0;
            }
            return v;
        })
        .collect();

    let mut thpp = vec![0.0; me];
    let mut thmp = vec![0.0; me];
    let thx_min = vec![0.0; me];
    let thx_max = funcao_igualdades_avaliadas
        .iter()
        .map(|&v| v.abs())
        .collect();

    let mut todos_possiveis_ds = Vec::new();
    let mut todos_possiveis_tgs = Vec::new();
    let mut todos_possiveis_thps = Vec::new();
    let mut todos_possiveis_thms = Vec::new();

    loop {
        // Restrição (1d)
        todos_possiveis_ds.push(dp.clone());
        dp = prox_ponto(dp.clone(), &d_min, &d_max, DIM, passo);
        if iguais(&dp, &d_min, DIM) {
            break;
        }
    }
    loop {
        // Restrição (1e)
        todos_possiveis_tgs.push(tgp.clone());
        tgp = prox_ponto(tgp.clone(), &tg_min, &tg_max, mi, passo);
        if iguais(&tgp, &tg_min, mi) {
            break;
        }
    }
    loop {
        // Restrição (1f)
        todos_possiveis_thps.push(thpp.clone());
        thpp = prox_ponto(thpp.clone(), &thx_min, &thx_max, me, passo);
        if iguais(&thpp, &thx_min, me) {
            break;
        }
    }
    loop {
        // Restrição (1g)
        todos_possiveis_thms.push(thmp.clone());
        thmp = prox_ponto(thmp.clone(), &thx_min, &thx_max, me, passo);
        if iguais(&thmp, &thx_min, me) {
            break;
        }
    }

    for d in todos_possiveis_ds {
        let restricao_1c_falha = false;
        // TODO: Restrição (1c)

        // calculo do lado esquerdo da expressão (1a)
        let mut lado_esq_1a = Vec::new();
        for idx in 0..mi {
            lado_esq_1a.push(
                funcao_desigualdades_avaliadas[idx]
                    + produto_interno(&grads_funcao_desigualdades[idx], &vec_arr_fixo(d.clone())),
            );
        }

        // calculo do lado esquerdo da expressão (1b)
        let mut lado_esq_1b = Vec::new();
        for idx in 0..me {
            lado_esq_1b.push(
                funcao_igualdades_avaliadas[idx]
                    + produto_interno(&grads_funcao_igualdades[idx], &vec_arr_fixo(d.clone())),
            );
        }

        if !restricao_1c_falha {
            for tg in &todos_possiveis_tgs {
                let mut restricao_1a_falha = false;

                for idx in 0..mi {
                    if !(lado_esq_1a[idx] <= tg[idx]) {
                        restricao_1a_falha = true;
                        break;
                    }
                }

                if !restricao_1a_falha {
                    for thp in &todos_possiveis_thps {
                        for thm in &todos_possiveis_thms {
                            let mut restricao_1b_falha = false;
                            for idx in 0..me {
                                if !(prox_o_suficiente_de_zero(
                                    lado_esq_1b[idx] - (thp[idx] - thm[idx]),
                                )) {
                                    restricao_1b_falha = true;
                                    break;
                                }
                            }

                            if !restricao_1b_falha {
                                let valor_funcao_objetivo: NumReal = produto_interno(
                                    &grad_funcao_objetivo,
                                    &vec_arr_fixo(d.clone()),
                                ) + C
                                    * (tg.iter().sum::<NumReal>()
                                        + thp.iter().sum::<NumReal>()
                                        + thm.iter().sum::<NumReal>());

                                pontos_viaveis.push((
                                    valor_funcao_objetivo,
                                    d.clone(),
                                    tg.clone(),
                                    thp.clone(),
                                    thm.clone(),
                                ));
                            }
                        }
                    }
                }
            }
        }
    }

    if pontos_viaveis.is_empty() {
        panic!("Não existe solução para o problema linear");
    }

    // procurando o menor em pontos viáveis
    let mut min_ponto_viavel = &pontos_viaveis[0];

    for ponto in &pontos_viaveis {
        if ponto.0 < min_ponto_viavel.0 {
            min_ponto_viavel = ponto;
        }
    }

    return min_ponto_viavel.clone();
}

// Função que resolve o subproblema linear avaliado em xⁱ
// Retorna os 4 vetores em ordem e o valor objetivo linear,
// onde:
// 1 - valor objetivo
// 2 - d
// 3 - tg
// 4 - th+
// 5 - th-
pub fn _resolver_problema_linear(
    problema: &Problema,
    x_i: Ponto,
) -> (
    NumReal,
    Vec<NumReal>,
    Vec<NumReal>,
    Vec<NumReal>,
    Vec<NumReal>,
) {
    // Biblioteca usada
    use minilp::*;

    // Calcula o gradiente da função objetivo em xⁱ
    // ∇f(xⁱ)
    let grad_funcao_objetivo = auto_grad(x_i, problema.funcao_objetivo);

    // Calcula os gradientes das me funções de igualdades em xⁱ
    // ∇hᵣ(xⁱ), r = 1, ..., me
    let grads_funcao_igualdades: Vec<Ponto> = problema
        .restricoes_igualdades
        .iter()
        .map(|&f| auto_grad(x_i, f))
        .collect();

    // Calcula os gradientes das mi funções de desigualdades em xⁱ
    // ∇gⱼ(xⁱ), j = 1, ..., mi
    let grads_funcao_desigualdades: Vec<Ponto> = problema
        .restricoes_desigualdades
        .iter()
        .map(|&f| auto_grad(x_i, f))
        .collect();

    // Calcula o valor de todas as mi funções de desigualdades em xⁱ
    // gⱼ(xⁱ), j = 1, ..., mi
    let funcao_desigualdades_avaliadas: Vec<NumReal> = problema
        .restricoes_desigualdades
        .iter()
        .map(|&f| f(x_i))
        .collect();

    // Calcula o valor de todas as me funções de igualdades em xⁱ
    // hᵣ(xⁱ), r = 1, ..., me
    let funcao_igualdades_avaliadas: Vec<NumReal> = problema
        .restricoes_igualdades
        .iter()
        .map(|&f| f(x_i))
        .collect();

    // Quantidade de variáveis
    let n = DIM; // Dimensão do vetor d (que também é a dimensão do dominio do problema)
    let mi = problema.restricoes_desigualdades.len(); // Dimensão do vetor tᵍ
    let me = problema.restricoes_igualdades.len(); // Dimensão dos vetores tʰ⁺ e tʰ⁻

    // Cria o problema de minimização
    let mut problema_minilp = Problem::new(OptimizationDirection::Minimize);

    // A função objetivo do subproblema linear é
    // ∇f(xⁱ)ᵀd + C(Σtᵍⱼ + Σtʰ⁺ᵣ Σtʰ⁻ᵣ)
    // ou ainda, deixando na forma de somas de produtos
    // ∇f(xⁱ)ᵀd + ΣC*tᵍⱼ + ΣC*tʰ⁺ᵣ ΣC*tʰ⁻ᵣ
    // A partir dai, sabendo que nossas variaveis são, d, tᵍⱼ, tʰ⁺ᵣ e tʰ⁻ᵣ,
    // podemos pegar os coeficientes de cada uma dessas variaveis

    // Cria o vetor c de coeficientes da função objetivo do subproblema linear
    let mut c: Vec<NumReal> = Vec::with_capacity(n + mi + 2 * me);

    // Os primeiros coeficientes (que são os coeficientes da direção d)
    // são os componentes do gradiente da função objetivo do subproblema linear
    for idx_componente_gradiente in 0..n {
        c.push(grad_funcao_objetivo[idx_componente_gradiente]);
    }

    // Depois são os coeficientes das variáveis t de relaxamento
    // Os coeficientes são todos C
    for _ in n..(n + mi + 2 * me) {
        c.push(C);
    }

    // Cria o vetor d de variáveis que vão ser otimizadas
    let mut d: Vec<Variable> = Vec::with_capacity(n);

    // Cria o vetor tg de variaveis que vão ser otimizadas
    let mut tg: Vec<Variable> = Vec::with_capacity(mi);

    // Cria o vetor th+ de variaveis que vão ser otimizadas
    let mut thp: Vec<Variable> = Vec::with_capacity(me);

    // Cria o vetor th- de variaveis que vão ser otimizadas
    let mut thm: Vec<Variable> = Vec::with_capacity(me);

    // As restrições (1d), (1e), (1f) e (1g) podem ser vistas como descrições
    // do domínio das variáveis, então, como a biblioteca permite, já se pode
    // impor essas restrições como domínios mesmo na criação da variável.

    // A definição da função objetivo do subproblema linear, é vista pela biblioteca
    // como sendo a criação das variáveis que devem ser otimizadas atribuídas à
    // seus coeficientes e domínios

    // As variáveis d são restritas em seu domínio e tem como coeficientes
    // os componentes do gradiente da função objetivo principal avaliada em xⁱ.
    for componente_d in 0..n {
        let coeficiente = c[componente_d];
        // As restrições que marcam o domínio de d são dₗ e dᵤ, como em (1d)
        let dominio = (problema.d_l[componente_d], problema.d_u[componente_d]);
        let var = problema_minilp.add_var(coeficiente, dominio);
        d.push(var);
    }

    // As variáveis tᵍ são restritas em seu domínio
    // e tem como coeficientes o valor C.
    for j in 0..mi {
        let coeficiente = C;

        // Restrição (1e)
        let dominio_fechado = (0.0, max(0.0, funcao_desigualdades_avaliadas[j]));
        let var = problema_minilp.add_var(coeficiente, dominio_fechado);
        tg.push(var);
    }

    // As variáveis  tʰ⁺ são restritas em seu domínio
    // e tem como coeficientes o valor C.
    for r in 0..me {
        let coeficiente = C;
        // Restrição (1f)
        let dominio_fechado = (0.0, funcao_igualdades_avaliadas[r].abs());
        let var = problema_minilp.add_var(coeficiente, dominio_fechado);
        thp.push(var);
    }

    // As variáveis  tʰ⁻ são restritas em seu domínio
    // e tem como coeficientes o valor C.
    for r in 0..me {
        let coeficiente = C;
        // Restrição (1g)
        let dominio_fechado = (0.0, funcao_igualdades_avaliadas[r].abs());
        let var = problema_minilp.add_var(coeficiente, dominio_fechado);
        thm.push(var);
    }

    // Por simplicidade, criar cada restrição restante separadamente

    // A biblioteca usada requer que a restrição esteja no seguinte formato:
    // (soma dos produtos das variaveis com respectivos coeficientes) (= / ≥ / ≤) (número real)
    // Então é necessário ajustar as restrições para que caibam na biblioteca.

    // (1a) gⱼ(xⁱ) + ∇gⱼ(xⁱ)ᵀd <= tᵍⱼ
    // Que ajustada fica
    // (1a) ∇gⱼ(xⁱ)ᵀd +(-tᵍ)ⱼ <= -gⱼ(xⁱ)
    for j in 0..mi {
        // Cria um expressão linear vazia
        let mut expressao_linear = LinearExpr::empty();

        // Como é um produto interno, os componentes do gradiente de gⱼ já são
        // os coeficientes das variáveis que precisam ser encontradas
        for (idx_grad, &componente_d) in d.iter().enumerate() {
            expressao_linear.add(componente_d, grads_funcao_desigualdades[j][idx_grad]);
        }

        // Adiciona o ultimo termo que é tᵍⱼ com coeficiente -1
        expressao_linear.add(tg[j], -1.0);

        // Adiciona a restrição
        problema_minilp.add_constraint(
            expressao_linear,
            ComparisonOp::Le,
            -funcao_desigualdades_avaliadas[j],
        );
    }

    // (1b) hᵣ(xⁱ) + ∇hᵣ(xⁱ)ᵀd = tʰ⁺ᵣ - tʰ⁻ᵣ
    // Que ajustada fica
    // (1b) ∇hᵣ(xⁱ)ᵀd + (-tʰ⁺ᵣ) + tʰ⁻ᵣ = - hᵣ(xⁱ)
    for r in 0..me {
        // Cria um expressão linear vazia
        let mut expressao_linear = LinearExpr::empty();

        // Como é um produto interno, os componentes do gradiente de hᵣ já são
        // os coeficientes das variáveis que precisam ser encontradas
        for (idx_grad, &componente_d) in d.iter().enumerate() {
            expressao_linear.add(componente_d, grads_funcao_igualdades[r][idx_grad]);
        }

        // Adiciona os dois últimos termos que são tʰ⁺ᵣ e tʰ⁻ᵣ com coeficientes -1 e 1 respectivamente
        expressao_linear.add(thp[r], -1.0);
        expressao_linear.add(thm[r], 1.0);

        // Adiciona a restrição
        problema_minilp.add_constraint(
            expressao_linear,
            ComparisonOp::Eq,
            -funcao_igualdades_avaliadas[r],
        );
    }

    // TODO: Restrição (1c)

    // Finalmente pede pra resolver o problema
    let solucao = problema_minilp.solve();

    // Verifica a solução
    match solucao {
        // Se a solução foi um sucesso, ok, salva as informações da solução em s
        Ok(s) => {
            // Imprime na tela em caso de debug
            if DEBUG {
                for (idx, &dc) in d.iter().enumerate() {
                    let strd = format!("d[{}] = {}", idx, s[dc]);
                    dbg!(strd);
                }

                for (idx, &tgc) in tg.iter().enumerate() {
                    let strd = format!("tg[{}] = {}", idx, s[tgc]);
                    dbg!(strd);
                }

                for (idx, &thpc) in thp.iter().enumerate() {
                    let strd = format!("thp[{}] = {}", idx, s[thpc]);
                    dbg!(strd);
                }

                for (idx, &thmc) in thm.iter().enumerate() {
                    let strd = format!("thm[{}] = {}", idx, s[thmc]);
                    dbg!(strd);
                }

                let strd = format!("min função objetivo = {}", s.objective());
                dbg!(strd);
            }

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
            let valor_funcao_objetivo =
                produto_interno(&vec_arr_fixo(ds.clone()), &grad_funcao_objetivo)
                    + C * (tgs.iter().sum::<NumReal>()
                        + thps.iter().sum::<NumReal>()
                        + thms.iter().sum::<NumReal>());

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
            // Imprime na tela em caso de debug
            if DEBUG {
                for (idx, &dc) in d.iter().enumerate() {
                    let strd = format!("d[{}] = {}", idx, s[dc]);
                    dbg!(strd);
                }

                for (idx, &tgc) in tg.iter().enumerate() {
                    let strd = format!("tg[{}] = {}", idx, s[tgc]);
                    dbg!(strd);
                }

                for (idx, &thpc) in thp.iter().enumerate() {
                    let strd = format!("thp[{}] = {}", idx, s[thpc]);
                    dbg!(strd);
                }

                for (idx, &thmc) in thm.iter().enumerate() {
                    let strd = format!("thm[{}] = {}", idx, s[thmc]);
                    dbg!(strd);
                }

                let strd = format!("min função objetivo = {}", s.objective());
                dbg!(strd);
                dbg!(s.clone());
            }

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

        for (idx, &var) in vars.iter().enumerate() {
            // Adiciona à expressão linear a variavel e seu coeficiente
            expr.add(var, at[idx][idx]);
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
            // Imprime na tela em caso de debug
            if DEBUG {
                for (idx, &v) in vars.iter().enumerate() {
                    let strd = format!("v[{}] = {}", idx, s[v]);
                    dbg!(strd);
                }

                let strd = format!("min função dual = {}", s.objective());
                dbg!(strd);
                dbg!(s.clone());
            }

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
