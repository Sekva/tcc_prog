mod defs;
mod emfcq;
mod estimativa_mul_lagrange;
mod funcao_merito;
mod iter_linear;
mod lagrangianas;
mod matricial;
mod ponto_estacionario;
mod prob_linear;
mod regiao_de_confianca;
mod utils;

use iter_linear::*;

use crate::{
    defs::{Funcao, Ponto, Problema, A, B, CC, D, DIM, E, F, L1, L2, O1, O2},
    emfcq::emfcq,
    funcao_merito::verificacao_funcao_merito,
    ponto_estacionario::checar_ponto_estacionario,
    regiao_de_confianca::verificar_regiao_de_confianca,
};

fn main() {
    // let funcao_objetivo: Funcao = |x: Ponto| (x[0].powi(2) - x[1].powi(2) - 1.0).sqrt();

    // f3: https://www.sfu.ca/~ssurjano/boha.html
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

    let x_inicial: Ponto = [2.0, 0.0];

    let mut p = Problema::novo(
        funcao_objetivo,
        restricoes_desigualdades,
        restricoes_igualdades,
        [-4.0, -4.0],
        [4.0, 4.0],
    );

    let problema_emfcq = emfcq(&p, 1.0);
    if problema_emfcq {
        // println!("EMFCQ? Sim");
    } else {
        // println!("EMFCQ? Não :/");
        std::process::exit(1);
    }

    let mut k = 1;
    let mut x = x_inicial;

    let mut x_novo;
    let mut d;
    let mut multiplicadores_de_lagrange;
    let mut verificacao_ponto_estacionario;
    let mut alpha;

    let mut hessiana_lagrangiana = vec![vec![0.0; DIM]; DIM];
    for i in 0..DIM {
        hessiana_lagrangiana[i][i] = 1.0;
    }

    let mut otimo: Option<Ponto> = None;

    while k < 100 {
        let resultado_iteracoes_lineares = iteracoes_lineares(&p, x, hessiana_lagrangiana.clone());

        x_novo = resultado_iteracoes_lineares.0;
        d = resultado_iteracoes_lineares.1;
        multiplicadores_de_lagrange = resultado_iteracoes_lineares.5;
        hessiana_lagrangiana = resultado_iteracoes_lineares.6;
        verificacao_ponto_estacionario = resultado_iteracoes_lineares.7;
        alpha = resultado_iteracoes_lineares.8;

        if verificacao_ponto_estacionario {
            otimo = Some(x_novo);
            println!("parada lp");
            break;
        }

        x_novo = verificacao_funcao_merito(&p, x_novo, x, alpha, d, &multiplicadores_de_lagrange);

        let (d_l, d_u) = verificar_regiao_de_confianca(&p, &x_novo, &x);
        p.atualizar_regiao_de_confianca(d_l, d_u);

        if checar_ponto_estacionario(&p, &x_novo, &multiplicadores_de_lagrange) {
            otimo = Some(x_novo);
            println!("parada nlp");
            break;
        }

        x = x_novo;
        k += 1;
    }

    match otimo {
        Some(ponto) => {
            println!("Otimo = {:?}", ponto)
        }
        _ => println!("Otimo não encontrado"),
    }
}
