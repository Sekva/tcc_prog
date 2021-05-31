use crate::defs::*;
use crate::utils::*;

// Função que verifica se as restrições passam nas qualificações estendidas de Mangassarian Fromovitz
// Se alguma condição for quebrada, retorna, se não, continua analisando até ter visto todo o espaço
// E então retorna que passa nas qualificações
// Como deve ser analisado em pontos discretos, incrementa os pontos de acordo com um dado passo
pub fn emfcq(problema: &Problema, passo: f64) -> bool {
    // Tem que analisar todo o espaço, nesse caso limitado, então começa do menor ponto possivel
    let mut x = problema.d_l;

    loop {
        // println!("x: {:?}", x);

        // Calcula os gradientes das funções de restrições de igualdades em x
        let grads_hr = problema
            .restricoes_igualdades
            .iter()
            .map(|restricao| auto_grad(x, restricao.clone()))
            .collect();

        // Caso alguma dupla de gradientes sejam dependentes linearmente, então a qualificação já foi quebrada
        if problema.restricoes_igualdades.len() > 1 {
            if sao_linearmente_dependentes(&grads_hr) {
                println!("Gradientes de h_r(x) são linearmente dependentes ");
                return false;
            }
        }

        // Agora o que falta é procurar um z para esse x que satisfaça as outras condições

        let mut z = problema.d_l; // Começa do menor possivel do espaço
        let mut existe_z = false; // Assumo que não existe um z, até encontrar um, ou não
        loop {
            //println!("x: {:?}, z: {:?}", x, z);

            // Se o produto interno entre algum gradiente e z for diferente (ou desconsideravel), procurar outro z que satisfaça
            for grad_hr in &grads_hr {
                if !prox_o_suficiente_de_zero(produto_interno(grad_hr, &z)) {
                    z = vec_arr_fixo(prox_ponto(
                        Vec::from(z),
                        &Vec::from(problema.d_l),
                        &Vec::from(problema.d_u),
                        DIM,
                        passo,
                    ));
                    continue;
                }
            }

            // Se chegou até aqui, então pra esse ser o z certo, basta que todos os produtos internos
            // entre os gradientes das funções de restrições de desigualdades que foram violadas, ou
            // que estão quase sendo violadas em x, e z sejam negativos. O que garente que existe uma
            // direção de descida para a restrição não ser violada ou se afasta da zona de fronteira
            // do violamento.
            let mut algum_gj_falha = false;
            for &gj in problema.restricoes_desigualdades.iter() {
                if gj(x) >= 0.0 {
                    // Só as funções que estão na fronteira ou que já foram violadas
                    let grad = auto_grad(x, gj);
                    if produto_interno(&grad, &z).is_sign_positive() {
                        // Se o produto interno é positivo, já falhou com a condição, então proximo passo
                        algum_gj_falha = true;
                        break;
                    }
                }
            }

            // Se nenhuma condição falhou, então temos um z para o x
            if !algum_gj_falha {
                existe_z = true;
                break; // Não precisa procurar outro z
            }

            // Mesma verificação que x para saber se todo o espaço já foi analisado
            z = vec_arr_fixo(prox_ponto(
                Vec::from(z),
                &Vec::from(problema.d_l),
                &Vec::from(problema.d_u),
                DIM,
                passo,
            ));
            if prox_o_suficiente_de_zero(dist(problema.d_l, z)) {
                break;
            }
        }

        // Caso não exista um z, mesmo depois de visitar todo o espaço, então não passou
        // nas qualificações
        if !existe_z {
            println!("Não existe z pra x: {:?}", x);
            return false;
        }

        // Caso tenha passado, vai ser analisado para o proximo x, a menos que já tenha visitado todo o espaço
        x = vec_arr_fixo(prox_ponto(
            Vec::from(x),
            &Vec::from(problema.d_l),
            &Vec::from(problema.d_u),
            DIM,
            passo,
        ));
        if prox_o_suficiente_de_zero(dist(problema.d_l, x)) {
            break;
        }
    }
    // Analisado todo o espaço, e tudo de acordo com as restrições, então passou nas qualificações
    true
}
