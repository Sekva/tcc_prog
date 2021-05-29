use crate::{
    defs::{Funcao, MultiplicadoresDeLagrange, NumReal, Ponto, Problema},
    utils::{auto_grad, line_search, max, produto_escalar, produto_interno, soma_pontos},
};

fn gerar_funcao_merito(
    problema: &Problema,
    multiplicadores: &MultiplicadoresDeLagrange,
) -> impl Fn(Ponto) -> NumReal {
    let funcao_obj = problema.funcao_objetivo.clone();
    let g = problema.restricoes_desigualdades.clone();
    let h = problema.restricoes_igualdades.clone();

    let lbds_maiores: Vec<NumReal> = multiplicadores
        .lambdas
        .iter()
        .map(|el| el.abs() + 0.2)
        .collect();

    let mus_maiores: Vec<NumReal> = multiplicadores
        .mus
        .iter()
        .map(|el| el.abs() + 0.2)
        .collect();

    return move |x: Ponto| -> NumReal {
        let val_obj = funcao_obj(x);

        let mut soma_gj = 0.0;
        for j in 0..lbds_maiores.len() {
            soma_gj += lbds_maiores[j] * max(g[j](x), 0.0);
        }

        let mut soma_hr = 0.0;
        for r in 0..mus_maiores.len() {
            soma_hr += mus_maiores[r] * h[r](x).abs();
        }

        val_obj + soma_gj + soma_hr
    };
}

fn derivada_direcional_g(direcao: Ponto, g: Funcao, x: Ponto) -> NumReal {
    let val: NumReal = g(x);

    if val >= 0.0 {
        let grad_g = move |x: Ponto| auto_grad(x, g);
        let prod: NumReal = produto_interno(&grad_g(x), &direcao);

        if val > 0.0 {
            return prod;
        }

        return max(prod, 0.0);
    }

    0.0
}

fn derivada_direcional_h(direcao: Ponto, h: Funcao, x: Ponto) -> NumReal {
    let val: NumReal = h(x);
    let grad_g = move |x: Ponto| auto_grad(x, h);
    let prod: NumReal = produto_interno(&grad_g(x), &direcao);

    if val > 0.0 {
        return prod;
    }

    if val == 0.0 {
        return prod.abs();
    }

    -1.0 * prod
}

fn gerar_derivada_direcional_funcao_merito(
    d: Ponto,
    problema: &Problema,
    multiplicadores: &MultiplicadoresDeLagrange,
) -> impl Fn(Ponto) -> NumReal {
    let lbds_maiores: Vec<NumReal> = multiplicadores
        .lambdas
        .iter()
        .map(|el| el.abs() + 0.2)
        .collect();

    let mus_maiores: Vec<NumReal> = multiplicadores
        .mus
        .iter()
        .map(|el| el.abs() + 0.2)
        .collect();

    let funcao_obj = problema.funcao_objetivo.clone();
    let g = problema.restricoes_desigualdades.clone();
    let h = problema.restricoes_igualdades.clone();

    let grad_funcao_obj = move |x: Ponto| auto_grad(x, funcao_obj);

    return move |x: Ponto| -> NumReal {
        let val_grad_funcao_obj = produto_interno(&grad_funcao_obj(x), &d);

        let mut val_grad_g_acumulado = 0.0;
        for j in 0..g.len() {
            val_grad_g_acumulado += lbds_maiores[j] * derivada_direcional_g(d, g[j], x);
        }

        let mut val_grad_h_acumulado = 0.0;
        for r in 0..h.len() {
            val_grad_h_acumulado += mus_maiores[r] * derivada_direcional_h(d, h[r], x);
        }

        val_grad_funcao_obj + val_grad_g_acumulado + val_grad_h_acumulado
    };
}

pub fn verificacao_funcao_merito(
    problema: &Problema,
    x_novo: Ponto,
    x_atual: Ponto,
    alpha: NumReal,
    d: Ponto,
    multiplicadores: &MultiplicadoresDeLagrange,
) -> Ponto {
    let funcao_merito = gerar_funcao_merito(problema, multiplicadores);

    let derivada_direcional_funcao_merito =
        gerar_derivada_direcional_funcao_merito(d, problema, multiplicadores);

    let merito_x_novo: NumReal = funcao_merito(x_novo);
    let merito_x_atual: NumReal = funcao_merito(x_atual);

    let derivada_direcional_merito_x_novo: NumReal = derivada_direcional_funcao_merito(x_novo);
    let derivada_direcional_merito_x_atual: NumReal = derivada_direcional_funcao_merito(x_atual);

    let condicao_4 = merito_x_novo <= merito_x_atual;

    let sigma_menor =
        (merito_x_novo - merito_x_atual) / (alpha * derivada_direcional_merito_x_atual);
    let condicao_5 = (sigma_menor > 0.0) && (sigma_menor < 1.0);

    let eta_maior = derivada_direcional_merito_x_novo / derivada_direcional_merito_x_atual;
    let condicao_6 = (eta_maior > sigma_menor) && (eta_maior < 1.0);

    if condicao_4 && condicao_5 && condicao_6 {
        return x_novo;
    }

    // println!("condicao 4 = {:?}", condicao_4);
    // println!("condicao 5 = {:?}, σ = {:?}", condicao_5, sigma_menor);
    // println!("condicao 6 = {:?}, η = {:?}", condicao_6, eta_maior);

    let alpha_novo = line_search(x_atual, d, &funcao_merito);

    // veio o mesmo ponto... que o novo
    let x_novo_novo = soma_pontos(x_atual, produto_escalar(alpha_novo, d));

    // println!("ponto novo novo = {:?}", x_novo_novo);
    // println!("ponto novo = {:?}", x_novo);

    return x_novo_novo;
}
