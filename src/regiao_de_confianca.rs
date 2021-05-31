use crate::{
    defs::{NumReal, Ponto, Problema, DELTA_DEC, DELTA_INC, DIM},
    utils::{max, produto_escalar},
};

// Verifica as regiões de confiança para d
// Caso a região seja muito grande, o passo tomado
// pode ser grande demais
// Caso seja muito pequena, a resolução dos problemas
// lineares podem ter problemas para encontrar soluções,
// ou ainda, passos muito pequenos podem ser tomados,
// fazendo o algoritmo demorar mais do que deveria
pub fn verificar_regiao_de_confianca(
    problema: &Problema,
    x_novo: &Ponto,
    x_velho: &Ponto,
) -> (Ponto, Ponto) {
    // Copias locais
    let mut d_l = problema.d_l;
    let mut d_u = problema.d_u;

    // Lista da razão maxima do movimento de cada componente
    // entre iterações não lineares
    let mut delta_l: Vec<NumReal> = Vec::new();
    for l in 0..DIM {
        // Diferença no componente
        let diferenca = x_novo[l] - x_velho[l];

        // Razões da diferença com os limites da região
        let diferenca_movimento_l = diferenca / d_l[l];
        let diferenca_movimento_u = diferenca / d_u[l];

        // Salva apenas a maior diferença
        delta_l.push(max(diferenca_movimento_l, diferenca_movimento_u));
    }

    // Busca a maior diferença
    let delta_max: NumReal = *delta_l
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    // Caso a diferença seja abaixo de um certo limite
    // as regiões são diminuidas de acordo com a diferença
    if delta_max < DELTA_DEC {
        let escalar = delta_max / DELTA_DEC;
        if escalar != 0.0 {
            d_u = produto_escalar(escalar, d_u);
            d_l = produto_escalar(escalar, d_l);
        }
    }

    // Caso a diferença seja acima de um certo limite
    // as regiões são crescidas de acordo com a diferença
    if delta_max > DELTA_INC {
        let escalar = 2.0 * delta_max;
        if escalar != 0.0 {
            d_u = produto_escalar(escalar, d_u);
            d_l = produto_escalar(escalar, d_l);
        }
    }

    // Retorna as copias locais dos limites para que seja atualizado.
    // Existe o caso as regiões sequer são atualizas.
    (d_l, d_u)
}
