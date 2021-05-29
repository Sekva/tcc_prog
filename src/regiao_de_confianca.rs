use crate::{
    defs::{NumReal, Ponto, Problema, DELTA_DEC, DELTA_INC, DIM},
    utils::{max, produto_escalar},
};

pub fn verificar_regiao_de_confianca(
    problema: &Problema,
    x_novo: &Ponto,
    x_velho: &Ponto,
) -> (Ponto, Ponto) {
    let mut d_l = problema.d_l;
    let mut d_u = problema.d_u;

    let mut delta_l: Vec<NumReal> = Vec::new();

    for l in 0..DIM {
        let diferenca = x_novo[l] - x_velho[l];

        let diferenca_movimento_l = diferenca / d_l[l];
        let diferenca_movimento_u = diferenca / d_u[l];

        delta_l.push(max(diferenca_movimento_l, diferenca_movimento_u));
    }

    let delta_max: NumReal = *delta_l
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    if delta_max < DELTA_DEC {
        let escalar = delta_max / DELTA_DEC;
        d_u = produto_escalar(escalar, d_u);
        d_l = produto_escalar(escalar, d_l);
    }
    if delta_max > DELTA_INC {
        let escalar = 2.0 * delta_max;
        d_u = produto_escalar(escalar, d_u);
        d_l = produto_escalar(escalar, d_l);
    }

    (d_l, d_u)
}
