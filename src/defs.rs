// Considerando isso como infinitesimal
pub const DBL_EPS: f64 = 1e-12;

// Dimensão do problema, R^DIM
pub const DIM: usize = 2;

// Constante de inviabilidade das iterações lineares
pub const C: NumReal = 1.0;

// Aliases de tipo, pra facilitar o entendimento
pub type NumReal = f64;
pub type Ponto = [NumReal; DIM];
pub type Funcao = fn(Ponto) -> NumReal; // o tipo Funcao é um ponteiro de uma função de recebe um Ponto e retorna um NumReal
