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

// Computa o resultado das iterações de subproblemas
// lineares
pub fn iteracoes_lineares(
    problema: &Problema,
    x: Ponto,
    hessiana: Vec<Vec<NumReal>>,
) -> (
    Ponto,                     // Ponto encontrado
    Ponto,                     // Direção de descida d
    Vec<NumReal>,              // Vetor de relaxamentos tg
    Vec<NumReal>,              // Vetor de relaxamentos th⁺
    Vec<NumReal>,              // Vetor de relaxamentos th⁻
    MultiplicadoresDeLagrange, // Estimativa dos multiplicadores de lagrange
    Vec<Vec<NumReal>>,         // Aproximação da Hessiana no ponto
    bool,                      // Encontrado ponto KKT estacionario
    NumReal, // Tamanho do passo tomado tomado na direção d para chegar ao ponto encontrado
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

        // Solução do problema primal e dual deve ser igual a menos de um
        // erro computacional
        assert!(quase_iguais(
            &Vec::from([solucao_dual.0]),
            &Vec::from([solucao_primal.0]),
            1
        ));

        // Extração da aproximação dos multiplicadores de lagrange da solução do problema dual
        multiplicadores_de_lagrange = extrair_multiplicadores_de_lagrange(problema, solucao_dual);
        // dbg!(&multiplicadores_de_lagrange);

        // Separa informações da solução do problema primal
        let (_obj, d, tg, thp, thm) = solucao_primal.clone();

        // Já tendo extraido os multiplicadores de lagrange, verifica se é um ponto KKT
        // estacionario
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

        // Gera a função lagrangiana penalizada a partir das informações
        let funcao_lagrangiana_penalizada =
            lagrangiana_penalizada(problema.clone(), multiplicadores_de_lagrange.clone());

        // Copia a salva a direção de descida encontrada
        let d_tmp = vec_arr_fixo(solucao_primal.1.clone());
        direcoes.push(d_tmp);

        // Faz uma busca em linha na direção de descida, encontrando um tamanho otimo para o passo
        alpha = line_search(ponto_atual, d_tmp, &funcao_lagrangiana_penalizada);

        // Passo que vai ser tomado
        let aidi = produto_escalar(alpha, d_tmp);

        // Faz uma copia do ponto atual e atualiza para o proximo ponto
        let ponto_anterior = ponto_atual;
        ponto_atual = soma_pontos(ponto_atual, aidi);

        // println!("ponto subiter lp = {:?}", ponto_atual);

        // Gera a funcção lagrangiana do problema, já sabendo dos multiplicadores
        let funcao_lagrangiana = lagrangiana(problema.clone(), multiplicadores_de_lagrange.clone());

        // Calcula a diferença entre os gradientes em cada ponto
        let g_i = auto_grad(ponto_anterior, &funcao_lagrangiana);
        let g_i1 = auto_grad(ponto_atual, &funcao_lagrangiana);
        let yi = subtracao_pontos(g_i1, g_i);

        // Usa a variação do gradiente e do ponto para calcular
        // a atualização da hessiana
        hessiana_atual = bfgs(hessiana_atual, aidi, yi);

        // Condições de parada das subiterações lineares

        // Condição de parada linear 1
        // Como se tem um comportamento ortogonalizante durante as subiterações
        // lineares, no maximo ocorre o numero de dimensões em iterações
        if i > DIM {
            // println!("condicao 1");
            break;
        }

        // Condição de parada linear 2
        // Se a direção de descida é desconsideravel
        if prox_o_suficiente_de_zero(norma(&d_tmp)) {
            // println!("condicao 2");
            break;
        }

        // Condição de parada linear 3
        // Para se todas as variavéis t de relaxamento não relaxam mais o problema
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
        // Se já está na segunda subiteração, e alguma variavel de ralaxamento
        // é positiva, para
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
        // Se o tamanho do passo a ser tomado é muito proximo a 1,
        // deve ser feita uma atualização no tamanho da região de confiança
        // e/ou uma atualização no ponto para que se estaja na vizinhaça
        // melhorada em relação a atual
        if prox_o_suficiente_de_zero(1.0 - alpha) {
            // println!("condicao 5");
            break;
        }

        i += 1;
    }

    // Extrai os dados até o momento e retorna
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
