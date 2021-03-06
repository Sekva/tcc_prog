use crate::{
    defs::{MultiplicadoresDeLagrange, NumReal, Ponto, Problema, DIM},
    utils::prox_o_suficiente_de_zero,
};

// Verifica se o ponto é kkt estacionario de acordo
// com os criterios descritos no artigo
pub fn checar_ponto_estacionario(
    problema: &Problema,
    x: &Ponto,
    multiplicadores_de_lagrange: &MultiplicadoresDeLagrange,
) -> bool {
    // Condição 4
    // Todos os lambdas devem ser positivos ou nulos.
    // Quando não nulos, mantem o sinal do valor das
    // restrições de desigualdades, que quando não
    // cumprida, aumenta o valor da função lagrangiana
    // quando o objetivo é minimizar
    for lbd_j in &multiplicadores_de_lagrange.lambdas {
        if *lbd_j < 0.0 {
            return false;
        }
    }

    // Para todas as outras restrições, é necessario ter
    // as informações do problema avalido no ponto em
    // questão
    let problema_avaliado_em_x = problema.avaliar_em(x.clone());

    // Condição 1
    // Todas as restrições de desigualdades
    // devem ter sido cumpridas
    for j in 0..problema.mi() {
        if problema_avaliado_em_x.2[j] > 0.0 {
            return false;
        }
    }

    // Condição 2
    // Todas as restrições de igualdades
    // devem ter sido cumpridas
    for j in 0..problema.me() {
        if problema_avaliado_em_x.3[j] != 0.0 {
            return false;
        }
    }

    // Condição 3
    // O produto da restrição de desigualdade e seu multiplicador
    // deve ser nulo, já que quando no otimo, a função lagrangiana
    // minimizada é apenas a função objetivo minimizada
    for j in 0..problema.mi() {
        let lbd_gjx = multiplicadores_de_lagrange.lambdas[j] * problema_avaliado_em_x.2[j];
        if lbd_gjx != 0.0 {
            return false;
        }
    }

    // Condição 5
    // Condição de ponto de extremo extendido
    // para todas as funções, objetivo e restrições

    // Gradientes avaliado no ponto
    let grad_fn_obj = problema_avaliado_em_x.1;
    let grads_gj = problema_avaliado_em_x.4;
    let grads_hr = problema_avaliado_em_x.5;

    // Somatorio dos gradientes restrições de desigualdades escalados
    // por deus respectivos lambdas
    let mut gj_acumulado: Vec<NumReal> = vec![0.0; DIM];
    for j in 0..problema.mi() {
        let grad = grads_gj[j];
        let lbd_gj = multiplicadores_de_lagrange.lambdas[j];
        for idx in 0..DIM {
            gj_acumulado[idx] += lbd_gj * grad[idx];
        }
    }

    // Somatorio dos gradientes restrições de igualdades escalados
    // por deus respectivos mus
    let mut hr_acumulado: Vec<NumReal> = vec![0.0; DIM];
    for r in 0..problema.me() {
        let grad = grads_hr[r];
        let mu_hr = multiplicadores_de_lagrange.mus[r];
        for idx in 0..DIM {
            hr_acumulado[idx] += mu_hr * grad[idx];
        }
    }

    // Soma das três parcelas, componente a componente
    let mut soma: Vec<NumReal> = vec![0.0; DIM];
    for idx in 0..DIM {
        soma[idx] = grad_fn_obj[idx] + gj_acumulado[idx] + hr_acumulado[idx];
    }

    // Todos os componentes devem ser nulos, a menos de um erro gerado
    // pelas computações
    for idx in 0..DIM {
        if !prox_o_suficiente_de_zero(soma[idx]) {
            return false;
        }
    }

    true
}
