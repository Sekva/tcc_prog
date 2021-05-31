mod defs;
mod emfcq;
mod estimativa_mul_lagrange;
mod funcao_merito;
mod instancias;
mod iter_linear;
mod lagrangianas;
mod matricial;
mod ponto_estacionario;
mod prob_linear;
mod regiao_de_confianca;
mod utils;

use iter_linear::*;

use crate::{
    defs::{NumReal, Ponto, DIM},
    emfcq::emfcq,
    funcao_merito::verificacao_funcao_merito,
    instancias::gerar_instancias,
    ponto_estacionario::checar_ponto_estacionario,
    regiao_de_confianca::verificar_regiao_de_confianca,
    utils::_iguais,
};

fn main() {
    // Itera sobre uma lista de instancias de problemas
    for problema in gerar_instancias() {
        // Cria uma copia mutavel do problema localmente
        let mut p = problema.clone();

        // Ignora verificação EMFCQ
        let passar_emfcq = true;

        if !passar_emfcq {
            let problema_emfcq = emfcq(&p, 10.0);
            if problema_emfcq {
                println!("EMFCQ? Sim");
            } else {
                println!("EMFCQ? Não");
                continue;
            }
        }

        // Lista de pontos em cada iteção não linear
        let mut passos_tomados: Vec<Vec<NumReal>> = Vec::new();

        // Contador de iterações não lineares
        let mut k = 1;

        // Variavel de armazenamento do ponto corrente
        let mut x = p.x_inicial;

        // Estado de cada iteração não linear

        // Ponto que possivelmente é melhor que o atual
        let mut x_novo;

        // Direção tomada para encontrar o ponto x_novo e tamanho do passo na direção
        let mut d;
        let mut alpha;

        // Armazenamento do pedido de parada nas subiterções lineares
        let mut verificacao_ponto_estacionario;

        // Armazenamento para os multiplicadores_de e matriz aproximada da função lagrangiana
        // Como é usado o método BFGS, que faz aproximações iteradas à hessiana. Usar a
        // identidade é o recomendado
        let mut multiplicadores_de_lagrange;
        let mut hessiana_lagrangiana = vec![vec![0.0; DIM]; DIM];
        for i in 0..DIM {
            hessiana_lagrangiana[i][i] = 1.0;
        }

        // Armazenamento do possivel otimo
        let mut otimo: Option<Ponto> = None;

        // Limite de 100 iterações não lineares
        while k < 100 {
            // Calcula e extrai as informações das subiterações lineares
            let resultado_iteracoes_lineares =
                iteracoes_lineares(&p, x, hessiana_lagrangiana.clone());

            x_novo = resultado_iteracoes_lineares.0;
            d = resultado_iteracoes_lineares.1;
            multiplicadores_de_lagrange = resultado_iteracoes_lineares.5;
            hessiana_lagrangiana = resultado_iteracoes_lineares.6;
            verificacao_ponto_estacionario = resultado_iteracoes_lineares.7;
            alpha = resultado_iteracoes_lineares.8;

            // Se foi encontrado um ponto kkt estacionario nas iterações lineares
            if verificacao_ponto_estacionario {
                otimo = Some(x_novo);
                println!("Parada subiteração linear");
                break;
            }

            // Verifica se o proximo ponto reduz suficientemente a função de mérido
            // Caso não reduza, um ponto diferente é retornado
            // Caso reduza, retorna o mesmo ponto entregue para a verificação
            x_novo =
                verificacao_funcao_merito(&p, x_novo, x, alpha, d, &multiplicadores_de_lagrange);

            // Comparando o movimento do ponto observado entre iteraões e
            // atualizando as regiões de confiança para a busca das direções
            // de acordo com esse movimento
            let (d_l, d_u) = verificar_regiao_de_confianca(&p, &x_novo, &x);
            p.atualizar_regiao_de_confianca(d_l, d_u);

            // Verifica se o novo ponto encontrado é um kkt estacionario
            if checar_ponto_estacionario(&p, &x_novo, &multiplicadores_de_lagrange) {
                otimo = Some(x_novo);
                println!("Parada iterção não linear");
                break;
            }

            // Verifica se passou-se duas iterações não lineares e o ponto não se moveu
            // Para o algoritmo, mesmo que não seja um ponto kkt estacionario
            if k > 2 {
                let x_teste = Vec::from(x_novo);
                let x_ant = Vec::from(passos_tomados[k - 2].clone()).clone();
                let x_ant2 = Vec::from(passos_tomados[k - 3].clone()).clone();

                // Verifica se o proximo, o atual e o anterior são iguais
                if _iguais(&x_teste, &x_ant, DIM) && _iguais(&x_teste, &x_ant2, DIM) {
                    otimo = Some(x_novo);
                    println!("Parada por passos repetidos--");
                    break;
                }
            }

            if false {
                println!("vvvvvvvvvvvvv");
                println!("Iterção NLP");
                println!("x =      {:?}", x);
                println!("f(x) =   {:?}", (p.funcao_objetivo)(x));
                println!("d =      {:?}", d);
                println!("x_novo = {:?}", x_novo);
                println!("^^^^^^^^^^^^^");
            }

            // Atualiza o ponto
            x = x_novo;

            // Armazena esse novo
            passos_tomados.push(Vec::from(x));

            // Proxima iteração
            k += 1;
        }

        // Verifica o otimo
        match otimo {
            Some(ponto) => {
                println!("x* = {:?}", ponto);
                println!("f(x*) = {:?}", (p.funcao_objetivo)(ponto));
                println!("\nx* real = {:?}", p.solucao);
                println!("real f(x*) = {:?}", (p.funcao_objetivo)(p.solucao.unwrap()));
            }
            _ => println!("Otimo não encontrado"),
        }

        println!();
        println!();
    }
}
