// Considerando isso como infinitesimal
pub const DBL_EPS: f64 = 1e-12;

// Dimensão do problema, R^DIM
pub const DIM: usize = 2;

// Constante de inviabilidade das iterações lineares
pub const C: NumReal = 1.0;

pub const DEBUG: bool = true;

pub const A: NumReal = 1.0;
pub const B: NumReal = 1.0;
pub const CC: NumReal = 1.0;
pub const D: NumReal = -0.4;
pub const E: NumReal = 2.7182818;
pub const F: NumReal = -9.0;

pub const O1: NumReal = 2.0;
pub const O2: NumReal = 2.0;

pub const L1: NumReal = 4.0;
pub const L2: NumReal = 4.0;

// Aliases de tipo, pra facilitar o entendimento
pub type NumReal = f64;
pub type Ponto = [NumReal; DIM];
pub type Funcao = fn(Ponto) -> NumReal; // o tipo Funcao é um ponteiro de uma função de recebe um Ponto e retorna um NumReal

// Estrutura do problema
pub struct Problema {
    pub funcao_objetivo: Funcao, // A função objetivo que quer ser minimizada
    pub restricoes_igualdades: Vec<Funcao>, // A lista das m_e funções restrições de igualdades, onde h_r(x) == 0, para r = 1, ..., m_e
    pub restricoes_desigualdades: Vec<Funcao>, // A lista das m_i funções restrições de desigualdades, onde g_j(x) == 0, para j = 1, ..., m_i

    // dL e dU são pontos que limitam a buscam de direções d no subproblema linear
    // mas antes de ser usado no SCP, podem ser usados como pontos que restringem o
    // espaço total do problema, e, no EMFCQ, procurar no R^n todo é impossível,
    // além de que a implementação da verificação do EMFCQ exige que existam limites fixados
    pub d_l: Ponto,
    pub d_u: Ponto,

    //Armazenamento dos estados
    pub xs_linear: Vec<Ponto>,
    pub ds_linear: Vec<Ponto>,
}
impl Problema {
    // Retorna um novo problema preenchido a partir do mínimo
    pub fn novo(
        funcao_objetivo: Funcao,
        restricoes_desigualdades: Vec<Funcao>,
        restricoes_igualdades: Vec<Funcao>,
    ) -> Problema {
        Problema {
            funcao_objetivo,
            restricoes_igualdades,
            restricoes_desigualdades,
            d_l: [-4.0, -4.0],
            d_u: [4.0, 4.0],
            xs_linear: Vec::new(),
            ds_linear: Vec::new(),
        }
    }
}
