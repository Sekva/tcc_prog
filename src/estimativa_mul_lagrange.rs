use crate::defs::*;

// Extrai os multiplicadores de lagrange a partir da
// solução dual do problema. Como dito em 2.1.1 no
// artigo sobre o SCP, os valores das variaveis duais
// do subproblema linear são usadas como estimadores
// dos multiplicadores de Lagrange para // o problema.
// São necessários para resolver a restrição 1c do
// subproblema linear e encontrar um passo otimo uma
// vez encontrada a direção de descida. É dito que são
// extraidos a partir da solução, mas não dito exatamente
// como. É mostrada uma forma na prova do lema 3 no artigo,
// onde são extraidos a partir das variaveis duais das
// restrições 1a e 1b do problema primal. Como a restrição
// 1b é na verdade duas, os valores duais são somados. Como
// a segunda restrição de 1b é invertida, temos que o valor
// para 1b é o valor para a primeira restrição derivida menos
// o segundo.
pub fn extrair_multiplicadores_de_lagrange(
    problema: &Problema,
    solucao_dual: (NumReal, Vec<NumReal>),
) -> (Vec<NumReal>, Vec<NumReal>) {
    // Cria listas para armazenar os valores
    let mut lambdas = Vec::new(); // λ
    let mut mups = Vec::new(); // μ⁺
    let mut muns = Vec::new(); // μ⁻
    let mut mus = Vec::new(); // μ = μ⁺ - μ⁻

    // Dimensões do problema
    let mi = problema.restricoes_desigualdades.len();
    let me = problema.restricoes_igualdades.len();

    for j in 0..mi {
        // Extrai λ das primeiras mi soluções do problema dual
        lambdas.push(solucao_dual.1[j]);
    }

    for r in 0..me {
        // Extrai μ⁺ das me soluções do problema dual, depois das primeias mi
        mups.push(solucao_dual.1[mi + r]);
    }

    for r in 0..me {
        // Extrai μ⁻ das me soluções do problema dual, depois das primeias mi+me
        muns.push(solucao_dual.1[mi + me + r]);
    }

    for (mup, mun) in mups.iter().zip(muns.iter()) {
        // Para cada par  (μ⁺, μ⁻) faz μ = μ⁺ - μ⁻
        mus.push(mup - mun);
    }

    // Retorna
    return (lambdas, mus);
}
