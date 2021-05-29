// Considerando isso como infinitesimal
pub const DBL_EPS: f64 = 1e-12;

// Dimensão do self, R^DIM
pub const DIM: usize = 2;

// Constante de inviabilidade das iterações lineares
pub const C: NumReal = 1.0;

// Constante de definição das ellipses das restrições de desigualdades
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

// Constante ρ
pub const RHO: NumReal = 0.7055;

// Constantes de aumento e decremento de regiões de confiança
pub const DELTA_INC: NumReal = 0.75;
pub const DELTA_DEC: NumReal = 0.25;

// Aliases de tipo, pra facilitar o entendimento
pub type NumReal = f64;
pub type Funcao = fn(Ponto) -> NumReal; // o tipo Funcao é um ponteiro de uma função de recebe um Ponto e retorna um NumReal

// Definição de um ponto
pub type Ponto = [NumReal; DIM];

// Estrutura do self
#[derive(Clone, Debug)]
pub struct Problema {
    pub funcao_objetivo: Funcao, // A função objetivo que quer ser minimizada
    pub restricoes_igualdades: Vec<Funcao>, // A lista das m_e funções restrições de igualdades, onde h_r(x) == 0, para r = 1, ..., m_e
    pub restricoes_desigualdades: Vec<Funcao>, // A lista das m_i funções restrições de desigualdades, onde g_j(x) == 0, para j = 1, ..., m_i

    // dL e dU são pontos que limitam a buscam de direções d no subself linear
    // mas antes de ser usado no SCP, podem ser usados como pontos que restringem o
    // espaço total do self, e, no EMFCQ, procurar no R^n todo é impossível,
    // além de que a implementação da verificação do EMFCQ exige que existam limites fixados
    pub d_l: Ponto,
    pub d_u: Ponto,
}

impl Problema {
    // Retorna um novo self preenchido a partir do mínimo
    pub fn novo(
        funcao_objetivo: Funcao,
        restricoes_desigualdades: Vec<Funcao>,
        restricoes_igualdades: Vec<Funcao>,
        d_l: Ponto,
        d_u: Ponto,
    ) -> Self {
        Self {
            funcao_objetivo,
            restricoes_igualdades,
            restricoes_desigualdades,
            d_l,
            d_u,
        }
    }

    pub fn avaliar_em(
        &self,
        x: Ponto,
    ) -> (
        NumReal,      // Função avaliada em x
        Ponto,        // Gradiente da função avaliada em x
        Vec<NumReal>, // Restrições de desigualdades avaliadas em x
        Vec<NumReal>, // Restrições de igualdades avaliadas em x
        Vec<Ponto>,   // Gradientes das funções de desigualdades avaliadas em x
        Vec<Ponto>,   // Gradientes das funções de igualdades avaliadas em x
    ) {
        use crate::utils::auto_grad;

        let val_funcao_objetivo = (self.funcao_objetivo)(x);
        let grad_funcao_objetivo = auto_grad(x, self.funcao_objetivo);

        let grads_funcao_igualdades: Vec<Ponto> = self
            .restricoes_igualdades
            .iter()
            .map(|&f| auto_grad(x, f))
            .collect();

        let grads_funcao_desigualdades: Vec<Ponto> = self
            .restricoes_desigualdades
            .iter()
            .map(|&f| auto_grad(x, f))
            .collect();

        let funcao_desigualdades_avaliadas: Vec<NumReal> = self
            .restricoes_desigualdades
            .iter()
            .map(|&f| f(x))
            .collect();

        let funcao_igualdades_avaliadas: Vec<NumReal> =
            self.restricoes_igualdades.iter().map(|&f| f(x)).collect();

        return (
            val_funcao_objetivo,
            grad_funcao_objetivo,
            funcao_desigualdades_avaliadas,
            funcao_igualdades_avaliadas,
            grads_funcao_desigualdades,
            grads_funcao_igualdades,
        );
    }

    pub fn mi(&self) -> usize {
        return self.restricoes_desigualdades.len();
    }

    pub fn me(&self) -> usize {
        return self.restricoes_igualdades.len();
    }

    pub fn atualizar_regiao_de_confianca(&mut self, d_l: Ponto, d_u: Ponto) {
        self.d_l = d_l;
        self.d_u = d_u;
    }
}
#[derive(Debug, Clone)]
pub struct MultiplicadoresDeLagrange {
    pub lambdas: Vec<NumReal>,
    pub mus: Vec<NumReal>,
}

#[allow(dead_code)]
pub enum OP {
    ADD,
    SUB,
}
