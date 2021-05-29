// use std::time::SystemTime;

use crate::{
    defs::{MultiplicadoresDeLagrange, NumReal, Ponto, Problema, DIM},
    estimativa_mul_lagrange::extrair_multiplicadores_de_lagrange,
    lagrangianas::{lagrangiana, lagrangiana_penalizada},
    matricial::matriz_e_vetores_problema_linear,
    ponto_estacionario::checar_ponto_estacionario,
    prob_linear::{resolver_problema_dual_matriz, resolver_problema_linear_matriz},
    utils::{
        auto_grad, bfgs, line_search, norma, produto_escalar, prox_o_suficiente_de_zero,
        quase_iguais, soma_pontos, subtracao_pontos, vec_arr_fixo,
    },
};

pub fn iteracoes_lineares(
    problema: &Problema,
    x: Ponto,
    hessiana: Vec<Vec<NumReal>>,
) -> (
    Ponto,
    Ponto,
    Vec<NumReal>,
    Vec<NumReal>,
    Vec<NumReal>,
    MultiplicadoresDeLagrange,
    Vec<Vec<NumReal>>,
    bool,
    NumReal,
) {
    let mut ponto_atual = x;
    let mut hessiana_atual = hessiana;
    let mut solucao_primal;
    let mut multiplicadores_de_lagrange: MultiplicadoresDeLagrange;
    let mut direcoes = Vec::new();
    let mut alpha;
    let mut i = 1;

    loop {
        // Vetores de coeficientes de um problema de minimização do seguinte tipo:
        // min cᵀx
        // s. a: a·x ≥ b
        let (matriz_a, vetor_b, vetor_c) =
            matriz_e_vetores_problema_linear(problema, ponto_atual, &direcoes, &hessiana_atual);

        // let ti = SystemTime::now();
        solucao_primal = resolver_problema_linear_matriz(problema, &matriz_a, &vetor_b, &vetor_c);
        // let tf = ti.elapsed().unwrap();
        // println!("Solução do problema linear no ponto {:?}: {:?}", ponto_atual, solucao_primal);
        // println!("Resolvido em: {}ns", tf.as_nanos());
        // println!("Resolvido em: {}s", tf.as_secs_f64());

        // let ti = SystemTime::now();
        let solucao_dual = resolver_problema_dual_matriz(&matriz_a, &vetor_b, &vetor_c);
        // let tf = ti.elapsed().unwrap();
        // println!("Solução do problema dual no ponto {:?}: {:?}", ponto_atual, solucao_dual);
        // println!("Resolvido em: {}ns", tf.as_nanos());
        // println!("Resolvido em: {}s", tf.as_secs_f64());
        // println!();

        assert!(quase_iguais(
            &Vec::from([solucao_dual.0]),
            &Vec::from([solucao_primal.0]),
            1
        ));

        multiplicadores_de_lagrange = extrair_multiplicadores_de_lagrange(problema, solucao_dual);
        // println!("{:?}\n", multiplicadores_de_lagrange);

        let (_obj, d, tg, thp, thm) = solucao_primal.clone();

        if checar_ponto_estacionario(problema, &ponto_atual, &multiplicadores_de_lagrange) {
            return (
                ponto_atual,
                vec_arr_fixo(d),
                tg,
                thp,
                thm,
                multiplicadores_de_lagrange,
                hessiana_atual,
                true,
                0.0,
            );
        }

        let funcao_lagrangiana_penalizada =
            lagrangiana_penalizada(problema.clone(), multiplicadores_de_lagrange.clone());

        let d_tmp = vec_arr_fixo(solucao_primal.1.clone());
        direcoes.push(d_tmp);

        alpha = line_search(ponto_atual, d_tmp, &funcao_lagrangiana_penalizada);

        let aidi = produto_escalar(alpha, d_tmp);

        let ponto_anterior = ponto_atual;
        ponto_atual = soma_pontos(ponto_atual, aidi);

        // println!("ponto subiter lp = {:?}", ponto_atual);

        let funcao_lagrangiana = lagrangiana(problema.clone(), multiplicadores_de_lagrange.clone());

        let g_i = auto_grad(ponto_anterior, &funcao_lagrangiana);
        let g_i1 = auto_grad(ponto_atual, &funcao_lagrangiana);
        let yi = subtracao_pontos(g_i1, g_i);

        hessiana_atual = bfgs(hessiana_atual, aidi, yi);

        // Condição de parada linear 1
        if i > DIM {
            // println!("condicao 1");
            break;
        }

        // Condição de parada linear 2
        if prox_o_suficiente_de_zero(norma(&d_tmp)) {
            // println!("condicao 2");
            break;
        }

        // Condição de parada linear 3
        let mut parar_na_cond_3 = true;
        for j in 0..problema.mi() {
            if tg[j] < problema.restricoes_desigualdades[j](ponto_atual) {
                parar_na_cond_3 = false;
                break;
            }
        }

        if parar_na_cond_3 {
            for r in 0..problema.me() {
                let val = problema.restricoes_igualdades[r](ponto_atual).abs();
                if thp[r] < val {
                    parar_na_cond_3 = false;
                    break;
                }
                if thm[r] < val {
                    parar_na_cond_3 = false;
                    break;
                }
            }
        }

        if parar_na_cond_3 {
            // println!("condicao 3");
            break;
        }

        // Condição de parada linear 4
        if i > 1 {
            let mut parar = false;
            for t in tg {
                if t > 0.0 {
                    parar = true;
                    break;
                }
            }

            for t in thp {
                if t > 0.0 {
                    parar = true;
                    break;
                }
            }

            for t in thm {
                if t > 0.0 {
                    parar = true;
                    break;
                }
            }

            if parar {
                // println!("condicao 4");
                break;
            }
        }

        // Condição de parada linear 5
        if prox_o_suficiente_de_zero(1.0 - alpha) {
            // println!("condicao 5");
            break;
        }

        i += 1;
    }

    let (_obj, d, tg, thp, thm) = solucao_primal;

    return (
        ponto_atual,
        vec_arr_fixo(d),
        tg,
        thp,
        thm,
        multiplicadores_de_lagrange,
        hessiana_atual,
        false,
        alpha,
    );
}
