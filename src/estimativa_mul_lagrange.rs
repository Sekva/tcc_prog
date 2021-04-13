use crate::defs::*;
use crate::utils::*;

// Como dito em 2.1.1 no artigo sobre o SCP, os valores
// das variaveis duais do subproblema linear são usadas
// como estimadores dos multiplicadores de Lagrange para
// o problema. São necessários para resolver a restrição
// (1c) do subproblema linear e encontrar um passo otimo
// uma vez encontrada a direção de descida.
// A forumalação do problema dual do subproblema linear
// é mostrada na prova do lema 3 do artigo do SCP.
// Retorna (valor maximo, λ, μ)
pub fn estimar_multiplicadores_lagrangianos(
    problema: &Problema,
    x_k: Ponto,
) -> (NumReal, Vec<NumReal>, Vec<NumReal>) {
    // O problema é encontrar os valores maximos de
    // λⱼ, μ⁺ᵣ e μ⁻ᵣ
    // em Σλⱼgⱼ(xᴷ) + Σμ⁺ᵣhᵣ(xᴷ) - Σμ⁻ᵣhᵣ(xᴷ)
    // ou em forma de somas de produtos
    // Σλⱼgⱼ(xᴷ) + Σμ⁺ᵣhᵣ(xᴷ) + Σμ⁻ᵣ(-hᵣ(xᴷ))
    // restrito à
    // Σλⱼ∇gⱼ(xᴷ) + Σμ⁺ᵣ∇hᵣ(xᴷ) - Σμ⁻ᵣ∇hᵣ(xᴷ) = - ∇f(xᴷ)
    // ou em forma de somas de produtos
    // Σλⱼ∇gⱼ(xᴷ) + Σμ⁺ᵣ∇hᵣ(xᴷ) + Σμ⁻ᵣ(-∇hᵣ(xᴷ)) = - ∇f(xᴷ)
    // λⱼ, μ⁺ᵣ e μ⁻ᵣ todos positivos

    // Calcula o gradiente da função objetivo do problema principal
    // em xᴷ
    // ∇f(xᴷ)
    let grad_funcao_objetivo = auto_grad(x_k, problema.funcao_objetivo);

    // Calcula os gradientes das me funções de igualdades em xᴷ
    // ∇hᵣ(xᴷ), r = 1, ..., me
    let grads_funcao_igualdades: Vec<Ponto> = problema
        .restricoes_igualdades
        .iter()
        .map(|&f| auto_grad(x_k, f))
        .collect();

    // Calcula os gradientes das mi funções de desigualdades em xᴷ
    // ∇gⱼ(xᴷ), j = 1, ..., mi
    let grads_funcao_desigualdades: Vec<Ponto> = problema
        .restricoes_desigualdades
        .iter()
        .map(|&f| auto_grad(x_k, f))
        .collect();

    // Calcula o valor de todas as mi funções de desigualdades em xᴷ
    // gⱼ(xᴷ), j = 1, ..., mi
    let funcao_desigualdades_avaliadas: Vec<NumReal> = problema
        .restricoes_desigualdades
        .iter()
        .map(|&f| f(x_k))
        .collect();

    // Calcula o valor de todas as me funções de igualdades em xᴷ
    // hᵣ(xᴷ), r = 1, ..., me
    let funcao_igualdades_avaliadas: Vec<NumReal> = problema
        .restricoes_igualdades
        .iter()
        .map(|&f| f(x_k))
        .collect();

    let n = DIM;
    let mi = problema.restricoes_desigualdades.len();
    let me = problema.restricoes_igualdades.len();

    // Biblioteca usada
    use minilp::*;

    // Cria o problema de minimização
    let mut problema_dual_minilp = Problem::new(OptimizationDirection::Maximize);

    // Cria o vetor λ de variaveis que vão ser otimizadas
    let mut lbd: Vec<Variable> = Vec::with_capacity(mi);

    // Cria o vetor μ⁺ de variaveis que vão ser otimizadas
    let mut mup: Vec<Variable> = Vec::with_capacity(me);

    // Cria o vetor μ⁻ de variaveis que vão ser otimizadas
    let mut mum: Vec<Variable> = Vec::with_capacity(me);

    for j in 0..mi {
        // Todos os λ podem estar entre 0 e ∞
        let dominio = (0.0, f64::INFINITY);
        // O coeficiente de cada λ é gⱼ(xᴷ)
        let var = problema_dual_minilp.add_var(funcao_desigualdades_avaliadas[j], dominio);
        lbd.push(var);
    }

    for r in 0..me {
        // Todos os μ⁺ podem estar entre 0 e ∞
        let dominio = (0.0, f64::INFINITY);
        // O coeficiente de cada μ⁺ é hᵣ(xᴷ)
        let var = problema_dual_minilp.add_var(funcao_igualdades_avaliadas[r], dominio);
        mup.push(var);
    }

    for r in 0..me {
        // Todos os μ⁻ podem estar entre 0 e ∞
        let dominio = (0.0, f64::INFINITY);
        // O coeficiente de cada μ⁻ é -hᵣ(xᴷ)
        let var = problema_dual_minilp.add_var(-funcao_igualdades_avaliadas[r], dominio);
        mum.push(var);
    }

    // Para as restrições, como os valores são vetores, podemos trabalhar
    // com uma expressão para componente do vetor. Sendo n restrições ao
    // invés de apenas uma com um vetor

    for i in 0..n {
        // Cria um expressão linear vazia
        let mut expressao_linear = LinearExpr::empty();

        // Primeiro somatorio
        // Cada λ tem como coeficiente o i-ésimo componente do grandiente de gⱼ(xᴷ)
        for j in 0..mi {
            expressao_linear.add(lbd[j], grads_funcao_desigualdades[j][i]);
        }

        // Segundo somatorio
        // Cada μ⁺ tem como coeficiente o i-ésimo componente do grandiente de hᵣ(xᴷ)
        for r in 0..me {
            expressao_linear.add(mup[r], grads_funcao_igualdades[r][i]);
        }

        // Cada μ⁻ tem como coeficiente o i-ésimo componente negativo do grandiente de hᵣ(xᴷ)
        for r in 0..me {
            expressao_linear.add(mum[r], -grads_funcao_igualdades[r][i]);
        }

        // Adiciona a restrição
        problema_dual_minilp.add_constraint(
            expressao_linear,
            ComparisonOp::Eq,
            -grad_funcao_objetivo[i],
        );
    }

    // Finalmente pede pra resolver o problema
    let solucao = problema_dual_minilp.solve();

    // Verifica a solução
    match solucao {
        // Se a solução foi um sucesso, ok, salva as informações da solução em s
        Ok(s) => {
            // Imprime na tela em caso de debug
            if DEBUG {
                for (idx, &lbdc) in lbd.iter().enumerate() {
                    let strd = format!("λ[{}] = {}", idx, s[lbdc]);
                    dbg!(strd);
                }

                for (idx, &mupc) in mup.iter().enumerate() {
                    let strd = format!("μ⁺[{}] = {}", idx, s[mupc]);
                    dbg!(strd);
                }

                for (idx, &mumc) in mum.iter().enumerate() {
                    let strd = format!("μ⁻[{}] = {}", idx, s[mumc]);
                    dbg!(strd);
                }

                let strd = format!("max função objetivo dual = {}", s.objective());
                dbg!(strd);
            }

            // Coleta as informações da solução

            // Vetores onde vão ser contidos λ e μ
            let mut lbd_final = Vec::new();
            let mut mu_final = Vec::new();

            // Coleta λ
            for &lbdc in &lbd {
                lbd_final.push(s[lbdc]);
            }

            // Coleta μ
            // No artigo, μ = μ⁺ - μ⁻
            for (&mupc, &mumc) in mup.iter().zip(mum.iter()) {
                mu_final.push(s[mupc] - s[mumc]);
            }

            // Calcula o valor maximizado da função objetivo do problema dual
            let mut valor_maximo = 0.0;

            for j in 0..mi {
                valor_maximo += lbd_final[j] * funcao_desigualdades_avaliadas[j];
            }

            for (idx, &mupc) in mup.iter().enumerate() {
                valor_maximo += s[mupc] * funcao_igualdades_avaliadas[idx];
            }

            for (idx, &mumc) in mum.iter().enumerate() {
                valor_maximo += s[mumc] * -funcao_igualdades_avaliadas[idx];
            }

            return (valor_maximo, lbd_final, mu_final);
        }

        // Caso a solução retorne um erro...
        Err(erro) => {
            println!("Problema dual do linear não tem solução ... {}", erro);

            // Encerra o programa, não tem mais sentido continuar
            std::process::exit(1);
        }
    }
}
