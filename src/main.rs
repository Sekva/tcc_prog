mod defs;
mod emfcq;
mod estimativa_mul_lagrange;
mod iter_linear;
mod lagrangianas;
mod matricial;
mod prob_linear;
mod utils;
use iter_linear::*;

use crate::{
    defs::{Funcao, Ponto, Problema, A, B, CC, D, DIM, E, F, L1, L2, O1, O2},
    emfcq::emfcq,
};

fn main() {
    // let funcao_objetivo: Funcao = |x: Ponto| (x[0].powi(2) - x[1].powi(2) - 1.0).sqrt();

    let funcao_objetivo: Funcao = |x: Ponto| {
        x[0].powi(2) + 2.0 * x[1].powi(2)
            - 0.3 * (3.0 * 3.1415926 * x[0] + 4.0 * 3.1415926 * x[1]).cos()
            + 0.3
    };

    let restricoes_igualdades: Vec<Funcao> = vec![|x: Ponto| x[1]];

    let restricoes_desigualdades: Vec<Funcao> = vec![
        // Fechar a caixinha toda
        |x: Ponto| x[0] + x[1] - 15.0,
        |x: Ponto| x[0] - x[1] - 15.0,
        |x: Ponto| -x[0] + x[1] - 15.0,
        |x: Ponto| -x[0] - x[1] - 15.0,
        |x: Ponto| {
            A * (x[0] - L1).powi(2) + B * (x[1] - L2).powi(2) - CC * (x[0] - L1) * (x[1] - L2)
                + D * (x[0] - L1)
                + E * (x[1] - L2)
                + F
        },
        |x: Ponto| {
            A * (x[0] - O1).powi(2)
                + B * (x[1] - O2).powi(2)
                + CC * (x[0] - O1) * (x[1] - O2)
                + D * (x[0] - O1)
                + E * (x[1] - O2)
                + F
        },
    ];

    let p = Problema::novo(
        funcao_objetivo,
        restricoes_desigualdades,
        restricoes_igualdades,
    );

    let problema_emfcq = emfcq(&p, 1.0);
    if problema_emfcq {
        println!("EMFCQ? Sim");
    } else {
        println!("EMFCQ? Não :/");
        std::process::exit(1);
    }

    let x: Ponto = [2.0, 0.0];

    let mut hessiana_lagrangiana = vec![vec![0.0; DIM]; DIM];
    for i in 0..DIM {
        hessiana_lagrangiana[i][i] = 1.0;
    }

    let (x, d, tg, thp, thm, multiplicadores_de_lagrange, hessiana_lagrangiana) =
        iteracoes_lineares(&p, x, hessiana_lagrangiana);
}
