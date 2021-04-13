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

use std::time::*;

// main né
fn main() {
    // let funcao_objetivo: Funcao = |x: Ponto| (x[0].powi(2) - x[1].powi(2) - 1.0).sqrt();

    let funcao_objetivo: Funcao = |x: Ponto| {
        x[0].powi(2) + 2.0 * x[1].powi(2)
            - 0.3 * (3.0 * 3.1415926 * x[0] + 4.0 * 3.1415926 * x[1]).cos()
            + 0.3
    };

    let restricoes_igualdades: Vec<Funcao> = vec![|x: Ponto| x[1]];

    let restricoes_desigualdades: Vec<Funcao> = vec![
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

    let problema_emfcq = true; //emfcq(&p);
    if problema_emfcq {
        println!("EMFCQ? Sim");
    } else {
        println!("EMFCQ? Não :/");
        std::process::exit(1);
    }

    let x: Ponto = [0.0, 0.0];
    println!("f({:?}) = {:?}", x, (p.funcao_objetivo)(x));
    println!("∇f({:?}) = {:?}", x, auto_grad(x, p.funcao_objetivo));

    println!();

    let ti = SystemTime::now();
    let mults = estimar_multiplicadores_lagrangianos(&p, x);
    let tf = ti.elapsed().unwrap();
    println!("Solução do problema dual: {:?}", mults);
    println!("Resolvido em: {}ns", tf.as_nanos());
    println!("Resolvido em: {}s", tf.as_secs_f64());

    println!();

    let ti = SystemTime::now();
    let solucao = resolver_problema_linear(&p, x);
    let tf = ti.elapsed().unwrap();
    println!("Solução do problema linear: {:?}", solucao);
    println!("Resolvido em: {}ns", tf.as_nanos());
    println!("Resolvido em: {}s", tf.as_secs_f64());
}
