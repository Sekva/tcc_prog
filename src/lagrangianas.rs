use crate::{
    defs::{MultiplicadoresDeLagrange, NumReal, Ponto, Problema, RHO},
    utils::max,
};

// Retorna a função lagrangiana como definida no artigo
pub fn lagrangiana(
    problema: Problema,
    multiplicadores: MultiplicadoresDeLagrange,
) -> impl Fn(Ponto) -> NumReal {
    move |x: Ponto| {
        // f(x)
        let f_x = (problema.funcao_objetivo)(x);

        // λⱼgⱼ(x)⁺
        let mut lbd_g_x = 0.0;
        for j in 0..multiplicadores.lambdas.len() {
            lbd_g_x += multiplicadores.lambdas[j] * problema.restricoes_desigualdades[j](x)
        }

        // μᵣhᵣ(x)
        let mut mu_h_x = 0.0;
        for r in 0..multiplicadores.mus.len() {
            mu_h_x += multiplicadores.mus[r] * problema.restricoes_igualdades[r](x);
        }

        f_x + lbd_g_x + mu_h_x
    }
}

// Retorna uma função semelhente à lagrangiana, chamada de
// L̃, que pode vista como uma forma "exagerada" da lagrangiana,
// uma lagrangiana penalizada.
pub fn lagrangiana_penalizada(
    problema: Problema,
    multiplicadores: MultiplicadoresDeLagrange,
) -> impl Fn(Ponto) -> NumReal {
    move |x: Ponto| {
        // f(x)
        let f_x = (problema.funcao_objetivo)(x);

        // λⱼgⱼ(x)⁺
        // gⱼ(x)⁺ = max(0, gⱼ(x))
        let mut lbd_g_x = 0.0;
        for j in 0..multiplicadores.lambdas.len() {
            lbd_g_x +=
                multiplicadores.lambdas[j] * max(0.0, problema.restricoes_desigualdades[j](x));
        }

        // μᵣ|hᵣ(x)|
        let mut mu_h_x = 0.0;
        for r in 0..multiplicadores.mus.len() {
            mu_h_x = (multiplicadores.mus[r] * problema.restricoes_igualdades[r](x)).abs();
        }

        // λⱼ(gⱼ(x)⁺)²
        let mut lbd_g_x_penalizado = 0.0;
        for j in 0..multiplicadores.lambdas.len() {
            lbd_g_x_penalizado += multiplicadores.lambdas[j]
                * (max(0.0, problema.restricoes_desigualdades[j](x))).powi(2);
        }

        // |μᵣ|(hᵣ(x))²
        let mut mu_h_x_penalizado = 0.0;
        for r in 0..multiplicadores.mus.len() {
            mu_h_x_penalizado +=
                multiplicadores.mus[r].abs() * (problema.restricoes_igualdades[r](x)).powi(2);
        }

        f_x + lbd_g_x + mu_h_x + (RHO * lbd_g_x_penalizado) + (RHO * mu_h_x_penalizado)
    }
}
