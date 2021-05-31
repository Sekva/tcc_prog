use crate::{
    defs::{
        Funcao, MultiplicadoresDeLagrange, NumReal, Ponto, Problema, ETA_MERITO, LAG_INC,
        SIGMA_MERITO,
    },
    utils::{auto_grad, line_search, max, produto_escalar, produto_interno, soma_pontos},
};

// Retorna a função de merito para os multiplicadores dados
fn gerar_funcao_merito(
    problema: &Problema,
    multiplicadores: &MultiplicadoresDeLagrange,
) -> impl Fn(Ponto) -> NumReal {
    // Copias das funções
    let funcao_obj = problema.funcao_objetivo.clone();
    let g = problema.restricoes_desigualdades.clone();
    let h = problema.restricoes_igualdades.clone();

    // Gera uma lista de λ̅ⱼ > λⱼ
    let lbds_maiores: Vec<NumReal> = multiplicadores
        .lambdas
        .iter()
        .map(|el| el.abs() + LAG_INC)
        .collect();

    // Gera uma lista de μ̅ᵣ > μᵣx
    let mus_maiores: Vec<NumReal> = multiplicadores
        .mus
        .iter()
        .map(|el| el.abs() + LAG_INC)
        .collect();

    // Retorna a função que toma um ponto e retorna um número real
    // movendo todas as copias locais para o escopo da função retornada
    return move |x: Ponto| -> NumReal {
        // f(x)
        let val_obj = funcao_obj(x);

        // λ̅ⱼg(x)⁺
        let mut soma_gj = 0.0;
        for j in 0..lbds_maiores.len() {
            soma_gj += lbds_maiores[j] * max(g[j](x), 0.0);
        }

        // μ̅ᵣ|hᵣ(x)|
        let mut soma_hr = 0.0;
        for r in 0..mus_maiores.len() {
            soma_hr += mus_maiores[r] * h[r](x).abs();
        }

        // Retorna a soma das três parcelas
        val_obj + soma_gj + soma_hr
    };
}

// Calcula o valor do componente positivo da derivada direcional de uma função g,
// restrição de desigualdade, em uma direção d
fn derivada_direcional_g(direcao: Ponto, g: Funcao, x: Ponto) -> NumReal {
    // Se g(x) > 0, retorna ∇g(x)ᵀd
    // Se g(x) = 0, retorna max(∇g(x)ᵀd, 0)
    // Se g(x) < 0, retorna 0

    let val: NumReal = g(x);

    if val >= 0.0 {
        let grad_g = move |x: Ponto| auto_grad(x, g); // Cria a função gradiente de g
        let prod: NumReal = produto_interno(&grad_g(x), &direcao);

        if val > 0.0 {
            return prod;
        }

        return max(prod, 0.0);
    }

    0.0
}

// Calcula o valor da derivada direcional de uma função h,
// restrição de igualdade, em uma direção d
fn derivada_direcional_h(direcao: Ponto, h: Funcao, x: Ponto) -> NumReal {
    // Se h(x) > 0, retorna ∇h(x)ᵀd
    // Se h(x) = 0, retorna |∇h(x)ᵀd|
    // Se h(x) < 0, retorna -(∇h(x)ᵀd)

    let val: NumReal = h(x);
    let grad_h = move |x: Ponto| auto_grad(x, h);
    let prod: NumReal = produto_interno(&grad_h(x), &direcao);

    if val > 0.0 {
        return prod;
    }

    if val == 0.0 {
        return prod.abs();
    }

    -1.0 * prod
}

// Gera a função a derivada direcional
fn gerar_derivada_direcional_funcao_merito(
    d: Ponto,
    problema: &Problema,
    multiplicadores: &MultiplicadoresDeLagrange,
) -> impl Fn(Ponto) -> NumReal {
    // Calcula os multiplicadores aumentados
    let lbds_maiores: Vec<NumReal> = multiplicadores
        .lambdas
        .iter()
        .map(|el| el.abs() + LAG_INC)
        .collect();

    let mus_maiores: Vec<NumReal> = multiplicadores
        .mus
        .iter()
        .map(|el| el.abs() + LAG_INC)
        .collect();

    // Copias locais das funções
    let funcao_obj = problema.funcao_objetivo.clone();
    let g = problema.restricoes_desigualdades.clone();
    let h = problema.restricoes_igualdades.clone();

    // Função gradiente da função objetivo
    let grad_funcao_obj = move |x: Ponto| auto_grad(x, funcao_obj);

    // Retorna a função DdM(x)
    return move |x: Ponto| -> NumReal {
        // ∇f(x)ᵀd
        // Valor do grandiente da função objetivo avaliada no ponto ()
        let val_grad_funcao_obj = produto_interno(&grad_funcao_obj(x), &d);

        //  λ̅ⱼDdgⱼ(x)⁺
        // Valores das derivadas direcionais de Ddgⱼ(x)⁺
        let mut val_grad_g_acumulado = 0.0;
        for j in 0..g.len() {
            val_grad_g_acumulado += lbds_maiores[j] * derivada_direcional_g(d, g[j], x);
        }

        // μ̅ᵣDd|hᵣ(x)|
        // Valores das derivadas direcionais de Dd|hᵣ(x)|
        let mut val_grad_h_acumulado = 0.0;
        for r in 0..h.len() {
            val_grad_h_acumulado += mus_maiores[r] * derivada_direcional_h(d, h[r], x);
        }

        val_grad_funcao_obj + val_grad_g_acumulado + val_grad_h_acumulado
    };
}

// Verifica se o novo ponto reduz suficientemente a função de mérito,
// e quando não reduz, um ponto diferente é retornado na mesma direção
// reduzido.
pub fn verificacao_funcao_merito(
    problema: &Problema,
    x_novo: Ponto,
    x_atual: Ponto,
    alpha: NumReal,
    d: Ponto,
    multiplicadores: &MultiplicadoresDeLagrange,
) -> Ponto {
    // Cria a função de mérito a partir do problema dos multiplicadores, e a derivada direcional da mesma
    let funcao_merito = gerar_funcao_merito(problema, multiplicadores);
    let derivada_direcional_funcao_merito =
        gerar_derivada_direcional_funcao_merito(d, problema, multiplicadores);

    // Calcula a função de mérito nos dois pontos, bem suas derivadas direcionais
    let merito_x_novo: NumReal = funcao_merito(x_novo);
    let merito_x_atual: NumReal = funcao_merito(x_atual);

    let derivada_direcional_merito_x_novo: NumReal = derivada_direcional_funcao_merito(x_novo);
    let derivada_direcional_merito_x_atual: NumReal = derivada_direcional_funcao_merito(x_atual);

    // Verifica se foi reduzida a função de mérito
    let condicao_4 = merito_x_novo <= merito_x_atual;

    // Verifica se o passo tomado desacelera em um certo ritmo em relação ao anterior
    let condicao_5 = (merito_x_novo - merito_x_atual)
        <= (SIGMA_MERITO * alpha * derivada_direcional_merito_x_atual);

    // Não permite que o tamanho do passo tomado seja desconsideravél em relação ao ritmo
    let condicao_6 =
        derivada_direcional_merito_x_novo >= ETA_MERITO * derivada_direcional_merito_x_atual;

    // Se todas as condições neste ponto foram satisfeitas, o ponto é aceitavel,
    // do contrario é buscando um novo na mesma direção
    if condicao_4 && condicao_5 && condicao_6 {
        return x_novo;
    }

    // println!("condicao 4 = {:?}", condicao_4);
    // println!("condicao 5 = {:?}, σ = {:?}", condicao_5, sigma_menor);
    // println!("condicao 6 = {:?}, η = {:?}", condicao_6, eta_maior);

    // O novo ponto basta ser o minimo do valor da função de mérito na direção
    // de descida d

    // Busca um novo tamanho de passo.
    // Existe o caso em que o novo ponto pode ser o mesmo, ou ainda
    // já se estar no minimo local da função de mérito.
    // É possivel resolver esse problema adicionando outras
    // verificações e buscas
    let alpha_novo = line_search(x_atual, d, &funcao_merito);

    // Gera o novo ponto
    let x_novo_novo = soma_pontos(x_atual, produto_escalar(alpha_novo, d));

    // println!("ponto novo novo = {:?}", x_novo_novo);
    // println!("ponto novo = {:?}", x_novo);

    // Retorna o novo ponto
    return x_novo_novo;
}
