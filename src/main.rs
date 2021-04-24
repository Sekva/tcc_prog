mod defs;
use defs::*;

mod utils;
use utils::*;

mod prob_linear;
use prob_linear::*;

mod emfcq;
use emfcq::*;

mod estimativa_mul_lagrange;
use estimativa_mul_lagrange::*;

mod matricial;
use matricial::*;

use std::time::*;

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

    let x: Ponto = [2.0, 1.0];
    // let x: Ponto = [0.0, 0.0];
    println!("f({:?}) = {:?}", x, (p.funcao_objetivo)(x));
    println!("∇f({:?}) = {:?}", x, auto_grad(x, p.funcao_objetivo));

    println!();

    // Vetores de coeficientes de um problema de minimização do seguinte tipo:
    // min cᵀx
    // s. a: a·x ≥ b
    let (matriz_a, vetor_b, vetor_c) = matriz_e_vetores_problema_linear(&p, x);

    let ti = SystemTime::now();
    let solucao_primal = resolver_problema_linear_matriz(&p, &matriz_a, &vetor_b, &vetor_c);
    let tf = ti.elapsed().unwrap();
    println!(
        "Solução do problema linear no ponto {:?}: {:?}",
        x, solucao_primal
    );
    println!("Resolvido em: {}ns", tf.as_nanos());
    println!("Resolvido em: {}s", tf.as_secs_f64());

    println!();

    let ti = SystemTime::now();
    let solucao_dual = resolver_problema_dual_matriz(&matriz_a, &vetor_b, &vetor_c);
    let tf = ti.elapsed().unwrap();
    println!(
        "Solução do problema dual no ponto {:?}: {:?}",
        x, solucao_dual
    );
    println!("Resolvido em: {}ns", tf.as_nanos());
    println!("Resolvido em: {}s", tf.as_secs_f64());

    assert!(iguais(
        &Vec::from([solucao_dual.0]),
        &Vec::from([solucao_primal.0]),
        1
    ));

    let multiplicadores_de_lagrange = extrair_multiplicadores_de_lagrange(&p, solucao_dual);
    println!("{:?}", multiplicadores_de_lagrange);
}
