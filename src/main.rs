type NumReal = f64;
type Ponto = [NumReal; DIM];
type Funcao = fn(Ponto) -> NumReal;

use std::convert::TryInto;

fn vec_arr_fixo<T>(v_in: Vec<T>) -> [T; DIM]
where
    T: Copy,
{
    let mut v = Vec::new();

    if v_in.len() < DIM {
        panic!("Vetor muito pequeno para transformar em array");
    }

    for i in 0..DIM {
        v.push(v_in[i]);
    }

    v.try_into().unwrap_or_else(|v: Vec<T>| {
        panic!(
            "Tentou converter um array vetor de tamanho {} em um array de tamanho {}",
            v.len(),
            DIM
        )
    })
}

pub fn auto_grad(x: Ponto, f: Funcao) -> Ponto {
    let mut xr = Vec::new();
    for i in 0..x.len() {
        let mut x1 = x.clone();
        let mut x2 = x.clone();
        x1[i] -= DBL_EPS;
        x2[i] += DBL_EPS;
        let y1 = f(x1.clone());
        let y2 = f(x2.clone());
        xr.push((y2 - y1) / (x2[i] - x1[i]));
    }
    return vec_arr_fixo(xr);
}

pub fn produto_interno(a: &Ponto, b: &Ponto) -> NumReal {
    let mut acc = 0.0;

    for idx in 0..DIM {
        acc += a[idx] * b[idx];
    }

    return acc;
}

fn norma(x: &Ponto) -> NumReal {
    let mut soma = 0.0;
    for i in x.iter() {
        soma += i * i;
    }
    soma.sqrt()
}

pub fn prox_o_suficiente_de_zero(n: NumReal) -> bool {
    (n - 0.0).abs() < DBL_EPS
}

pub fn normalizar(ponto: &Ponto) -> Ponto {
    let norma = norma(ponto);
    let mut novo_povo = ponto.clone();

    for idx in 0..DIM {
        novo_povo[idx] /= norma;
    }

    novo_povo
}

pub fn dist(p: Ponto, q: Ponto) -> NumReal {
    let mut acc: f64 = 0.0;
    for i in 0..DIM {
        acc += (p[i] - q[i]).powi(2);
    }
    acc.sqrt()
}

pub fn sao_linearmente_dependentes(pontos: &Vec<Ponto>) -> bool {
    let normais: Vec<Ponto> = pontos.iter().map(|ponto| normalizar(ponto)).collect();

    for normal_idx in 0..normais.len() {
        for normal_idx_2 in 0..normais.len() {
            if normal_idx != normal_idx_2 {
                if prox_o_suficiente_de_zero(dist(normais[normal_idx], normais[normal_idx_2])) {
                    return true;
                }
            }
        }
    }

    false
}

pub struct Problema {
    pub funcao_objetivo: Funcao,
    pub restricoes_igualdades: Vec<Funcao>,
    pub restricoes_desigualdades: Vec<Funcao>,
    pub x_min: Ponto,
    pub x_max: Ponto,
}

impl Problema {
    pub fn prox_ponto(&self, mut x: Ponto, passo: NumReal) -> Ponto {
        x[x.len() - 1] += passo;

        for idx in (0..DIM).rev() {
            if x[idx] > self.x_max[idx] {
                x[idx] = self.x_min[idx];

                if idx > 0 {
                    x[idx - 1] += passo;
                }
            }
        }

        return x;
    }

    pub fn emfcq(&self) -> bool {
        let mut x = self.x_min;
        let passo = 0.1;

        loop {
            //println!("x: {:?}", x);
            let mut olhou_todo_espaco = true;
            for idx in 0..DIM {
                if x[idx] < self.x_max[idx] {
                    olhou_todo_espaco = false;
                }
            }

            if olhou_todo_espaco {
                break;
            }
            let grads_hr = self
                .restricoes_igualdades
                .iter()
                .map(|restricao| auto_grad(x, restricao.clone()))
                .collect();

            if self.restricoes_igualdades.len() > 1 {
                if sao_linearmente_dependentes(&grads_hr) {
                    println!("Gradientes de h_r(x) são linearmente dependentes ");
                    return false;
                }
            }

            //procurar um z

            let mut z = self.x_min;
            let mut existe_z = false;
            loop {
                //println!("x: {:?}, z: {:?}", x, z);
                let mut olhou_todas_as_opcoes_de_z = true;
                for idx in 0..DIM {
                    if z[idx] < self.x_max[idx] {
                        olhou_todas_as_opcoes_de_z = false;
                    }
                }
                if olhou_todas_as_opcoes_de_z {
                    break;
                }

                for grad_hr in &grads_hr {
                    if !prox_o_suficiente_de_zero(produto_interno(grad_hr, &z)) {
                        z = self.prox_ponto(z, passo);
                        continue;
                    }
                }

                let mut algum_gj_falha = false;
                for &gj in self.restricoes_desigualdades.iter() {
                    if gj(x) >= 0.0 {
                        let grad = auto_grad(x, gj);
                        if produto_interno(&grad, &z).is_sign_positive() {
                            algum_gj_falha = true;
                        }
                    }
                }

                if !algum_gj_falha {
                    //entao z existe, vamo ver o proximo x
                    existe_z = true;
                    break;
                }

                z = self.prox_ponto(z, passo);
            }

            if !existe_z {
                println!("Não existe z pra x: {:?}", x);
                return false;
            }

            x = self.prox_ponto(x, passo);
            if prox_o_suficiente_de_zero(dist(self.x_min, x)) {
                // já resetou
                break;
            }
        }

        true
    }
}

const DBL_EPS: f64 = 1e-12;
const DIM: usize = 2;

fn main() {
    let p = Problema {
        funcao_objetivo: |x: Ponto| x[0] * x[1],
        restricoes_igualdades: vec![|x: Ponto| x[0] + x[1] - 4.0],
        restricoes_desigualdades: vec![
            |x: Ponto| (x[0] - 2.0).powi(2) + (x[1] - 2.0).powi(2) - 2.0,
            |x: Ponto| (x[0] - 2.5).powi(2) + (x[1] - 2.0).powi(2) - 2.0,
        ],
        x_min: [-4.0, -4.0],
        x_max: [4.0, 4.0],
    };

    let problema_emfcq = p.emfcq();
    if problema_emfcq {
        println!("EMFCQ? Sim");
    } else {
        println!("EMFCQ? Não :/");
        std::process::exit(1);
    }

    let _x: Ponto = [3.0, 2.0];

    let _ = p.restricoes_igualdades;
    let _ = p.restricoes_desigualdades;
    let _ = p.x_min;
    let _ = p.x_max;
}
