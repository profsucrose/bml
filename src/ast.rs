use lasso::Spur;
use std::collections::HashMap;

use crate::logger::{report, ErrorType};

#[derive(Debug, Copy, Clone)]
pub enum Op {
    Add,
    Sub,
    Div,
    Mul,
    More,
    Less,
    MoreEq,
    LessEq,
    Equal,
}

#[derive(Copy, Clone, Debug)]
pub enum Field {
    X,
    Y,
    Z,
    W,
}

#[derive(Copy, Clone, Debug)]
pub struct Swizzle(
    pub Field,
    pub Option<Field>,
    pub Option<Field>,
    pub Option<Field>,
);

#[derive(PartialEq, Clone, Copy, Debug)]
pub enum Val {
    Float(f32),
    Vec2(f32, f32),
    Vec3(f32, f32, f32),
    Vec4(f32, f32, f32, f32),
}

use Val::*;

mod math {
    pub fn rad(x: f32) -> f32 {
        x * std::f32::consts::PI / 180.0
    }

    pub fn deg(x: f32) -> f32 {
        x * 180.0 / std::f32::consts::PI
    }

    pub fn inv_sqrt(x: f32) -> f32 {
        let i = x.to_bits();
        let i = 0x5f3759df - (i >> 1);
        let y = f32::from_bits(i);
    
        y * (1.5 - 0.5 * x * y * y)
    }

    pub fn mix(x: f32, y: f32, a: f32) -> f32 {
        x * (1.0 - a) + y * a
    }
}

impl Val {
    pub fn map<F: FnMut(f32) -> f32>(self, mut f: F) -> Val {
        match self {
            Float(x) => Float(f(x)),
            Vec2(x, y) => Vec2(f(x), f(y)),
            Vec3(x, y, z) => Vec3(f(x), f(y), f(z)),
            Vec4(x, y, z, w) => Vec4(f(x), f(y), f(z), f(w)),
        }
    }

    pub fn zipmap3<F: FnMut(f32, f32, f32) -> f32>(self, o1: Self, o2: Self, mut f: F) -> Val {
        match (self, o1, o2) {
            (Float(x), Float(y), Float(z)) => Float(f(x, y, z)),
            (Vec2(x0, x1), Vec2(y0, y1), Vec2(z0, z1)) => Vec2(f(x0, y0, z0), f(x1, y1, z1)),
            (Vec3(x0, x1, x2), Vec3(y0, y1, y2), Vec3(z0, z1, z2)) => Vec3(f(x0, y0, z0), f(x1, y1, z1), f(x2, y2, z2)),
            (Vec4(x0, x1, x2, x3), Vec4(y0, y1, y2, y3), Vec4(z0, z1, z2, z3)) => Vec4(f(x0, y0, z0), f(x1, y1, z1), f(x2, y2, z2), f(x3, y3, z3)),
            _ => panic!("{}", format!("expected all args to zipmap3 be same type, got {:?}, {:?}, {:?}", self, o1, o2))
        }
    }

    pub fn zipmap<F: FnMut(f32, f32) -> f32>(self, o: Self, mut f: F) -> Val {
        match (self, o) {
            (Float(l), Float(r)) => Float(f(l, r)),
            (Vec2(lx, ly), Float(r)) => Vec2(f(lx, r), f(ly, r)),
            (Vec3(lx, ly, lz), Float(r)) => Vec3(f(lx, r), f(ly, r), f(lz, r)),
            (Vec4(lx, ly, lz, lw), Float(r)) => Vec4(f(lx, r), f(ly, r), f(lz, r), f(lw, r)),

            (Vec2(lx, ly), Vec2(rx, ry)) => Vec2(f(lx, rx), f(ly, ry)),
            (Vec3(lx, ly, lz), Vec3(rx, ry, rz)) => Vec3(f(lx, rx), f(ly, ry), f(lz, rz)),
            (Vec4(lx, ly, lz, lw), Vec4(rx, ry, rz, rw)) => {
                Vec4(f(lx, rx), f(ly, ry), f(lz, rz), f(lw, rw))
            }
            _ => panic!("unsupported vector/scalar binary operation relationship"),
        }
    }

    pub fn get_field(&self, f: Field) -> f32 {
        use Val::*;
        match f {
            Field::X => match *self {
                Float(_) => panic!("Float has no x field"),
                Vec2(x, _) => x,
                Vec3(x, _, _) => x,
                Vec4(x, _, _, _) => x,
            },
            Field::Y => match *self {
                Float(_) => panic!("Float has no y field"),
                Vec2(_, y) => y,
                Vec3(_, y, _) => y,
                Vec4(_, y, _, _) => y,
            },
            Field::Z => match *self {
                Float(_) => panic!("Float has no z field"),
                Vec2(_, _) => panic!("Vec2 has no z field"),
                Vec3(_, _, z) => z,
                Vec4(_, _, z, _) => z,
            },
            Field::W => match *self {
                Float(_) => panic!("Float has no w field"),
                Vec2(_, _) => panic!("Vec2 has no w field"),
                Vec3(_, _, _) => panic!("Vec3 has no w field"),
                Vec4(_, _, _, w) => w,
            },
        }
    }

    pub fn radians(&self) -> Val { self.map(math::rad) }
    pub fn degrees(&self) -> Val { self.map(math::deg) }
    pub fn sin(&self) -> Val { self.map(f32::sin) }
    pub fn cos(&self) -> Val { self.map(f32::cos) }
    pub fn tan(&self) -> Val { self.map(f32::tan) }
    pub fn asin(&self) -> Val { self.map(f32::asin) }
    pub fn acos(&self) -> Val { self.map(f32::acos) }
    pub fn atan(&self) -> Val { self.map(f32::atan) }
    pub fn exp(&self) -> Val { self.map(f32::exp) }
    pub fn log(&self) -> Val { self.map(|x| f32::log(std::f32::consts::E, x)) }
    pub fn sqrt(&self) -> Val { self.map(f32::sqrt) }
    pub fn invsqrt(&self) -> Val { self.map(math::inv_sqrt) }
    pub fn abs(&self) -> Val { self.map(f32::abs) }
    pub fn sign(&self) -> Val { self.map(f32::signum) }
    pub fn floor(&self) -> Val { self.map(f32::floor) }
    pub fn ceil(&self) -> Val { self.map(f32::ceil) }
    pub fn fract(&self) -> Val { self.map(f32::fract) }

    pub fn dot(&self, o: Self) -> Result<Val, String> {
        match (self, o) {
            (Vec2(x0, y0), Vec2(x1, y1)) => Ok(Float(x0 * x1 + y0 * y1)),
            (Vec3(x0, y0, z0), Vec3(x1, y1, z1)) => Ok(Float(x0  * x1 + y0 * y1 + z0 * z1)),
            (Vec4(x0, y0, z0, w0), Vec4(x1, y1, z1, w1)) => Ok(Float(x0 * x1 + y0 * y1 + z0 * z1 + w0 + w1)),
            _ => Err(format!("Expected dot(vec2, vec2), dot(vec3, vec3), dot(vec4, vec4), got dot({:?}, {:?})", self, o))
        }
    }

    pub fn modulo(&self, o: Self) -> Result<Val, String> { 
        match (self, o) {
            (Float(_), Float(_)) 
                | (Vec2(_, _), Vec2(_, _))
                | (Vec3(_, _, _), Vec3(_, _, _))
                | (Vec4(_, _, _, _), Vec4(_, _, _, _))
            => Ok(self.zipmap(o, |x, y| x % y)),
            _ => Err(format!("Expected mod(float, float), mod(vec2, vec2), mod(vec3, vec3), mod(vec4, vec4), got mod({:?}, {:?})", self, o))
        }
    }

    pub fn min(&self, val: &Val) -> Result<Val, String> { 
        match (self, val) {
            (Float(_), Float(_)) 
                | (Vec2(_, _), Vec2(_, _))
                | (Vec3(_, _, _), Vec3(_, _, _))
                | (Vec4(_, _, _, _), Vec4(_, _, _, _))
            => Ok(self.zipmap(*val, |x, y| x.min(y))),
            _ => Err(format!("Expected min(float, float), min(vec2, vec2), min(vec3, vec3), min(vec4, vec4), got min({:?}, {:?})", self, val))
        }
    }

    pub fn max(&self, val: &Val) -> Result<Val, String> { 
        match (self, val) {
            (Float(_), Float(_)) 
                | (Vec2(_, _), Vec2(_, _))
                | (Vec3(_, _, _), Vec3(_, _, _))
                | (Vec4(_, _, _, _), Vec4(_, _, _, _))
            => Ok(self.zipmap(*val, |x, y| x.max(y))),
            _ => Err(format!("Expected max(float, float), max(vec2, vec2), max(vec3, vec3), max(vec4, vec4), got max({:?}, {:?})", self, val))
        }
    }

    pub fn clamp(&self, min: Self, max: Self) -> Result<Val, String> {
        match (self, min, max) {
            (Float(_), Float(_), Float(_)) 
                | (Vec2(_, _), Vec2(_, _), Vec2(_, _))
                | (Vec3(_, _, _), Vec3(_, _, _), Vec3(_, _, _))
                | (Vec4(_, _, _, _), Vec4(_, _, _, _), Vec4(_, _, _, _))
            => Ok(self.zipmap3(min, max, |x, y, z| x.clamp(y, z))),

            (Vec2(_, _), Vec2(_, _), Float(a))
                | (Vec3(_, _, _), Vec3(_, _, _), Float(a))
                | (Vec4(_, _, _, _), Vec4(_, _, _, _), Float(a))
            => Ok(self.zipmap3(min, Vec4(a, a, a, a), |x, y, z| x.clamp(y, z))), 

            _ => Err(format!("Expected clamp(float, float, float), clamp(vec2, vec2, vec2), clamp(vec3, vec3, vec3), clamp(vec4, vec4, vec4), got clamp({:?}, {:?}, {:?})", self, min, max))
        }
    }

    pub fn mix(&self, y: Self, a: Self) -> Result<Val, String> {
        match (self, y, a) {
            (Float(_), Float(_), Float(_)) 
                | (Vec2(_, _), Vec2(_, _), Vec2(_, _))
                | (Vec3(_, _, _), Vec3(_, _, _), Vec3(_, _, _))
                | (Vec4(_, _, _, _), Vec4(_, _, _, _), Vec4(_, _, _, _))
            => Ok(self.zipmap3(y, a, math::mix)),

            (Vec2(_, _), Vec2(_, _), Float(a))
                | (Vec3(_, _, _), Vec3(_, _, _), Float(a))
                | (Vec4(_, _, _, _), Vec4(_, _, _, _), Float(a))
            => Ok(self.zipmap3(y, Vec4(a, a, a, a), math::mix)),

            _ => Err(format!("Expected mix(float, float, float), mix(vec2, vec2, float), mix(vec3, vec3, float), mix(vec4, vec4, float), mix(vec2, vec2, vec2), mix(vec3, vec3, vec3), mix(vec4, vec4, vec4), got mix({:?}, {:?}. {:?}", self, y, a))
        }
    }

    pub fn step(&self, x: Self) -> Result<Val, String> {
        match (self, x) {
            (Float(_), Float(_)) 
                | (Vec2(_, _), Vec2(_, _)) 
                | (Vec3(_, _, _), Vec3(_, _, _)) 
                | (Vec4(_, _, _, _), Vec4(_, _, _, _)) 
            => Ok(self.zipmap(x, |edge, x| if x < edge { 0.0 } else { 1.0 })),

            (Float(a), Vec2(_, _)) 
                | (Float(a), Vec3(_, _, _)) 
                | (Float(a), Vec4(_, _, _, _)) 
            => Ok(x.map(|x| if x < *a { 0.0 } else { 1.0 })),

            _ => Err(format!("Expected step(float, float), step(float, vec2), step(float, vec3), step(float, vec4), step(vec2, vec2), step(vec3, vec3), step(vec4, vec4), got step({:?}, {:?})", self, x))
        }
    }

    pub fn length(&self) -> Val {
        let sum = match self.map(|x| x.powi(2)) {
            Float(x) => x,
            Vec2(x, y) => x + y,
            Vec3(x, y, z) => x + y + z,
            Vec4(x, y, z, w) => x + y + z + w
        };

        Float(sum.sqrt())
    }

    pub fn cross(&self, o: Self) -> Result<Val, String> {
        match (self, o) {
            (Vec3(x0, y0, z0), Vec3(x1, y1, z1)) => Ok(
                Vec3(
                    y0 * z1 - z0 * y1,
                    z0 * x1 - x0 * z1,
                    x0 * y1 - y0 * x1
                )
            ),
            _ => Err(format!("Expected cross(vec3, vec3), got cross({:?}, {:?})", self, o))
        }
    }

    pub fn norm(&self) -> Val {
        self.zipmap(self.length(), |x, len| x / len)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum BuiltIn {
    Dist,
    Radians,
    Degrees,
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Pow,
    Exp,
    Log,
    Sqrt,
    InverseSqrt,
    Abs,
    Sign,
    Floor,
    Ceil,
    Fract,
    Mod,
    Min,
    Max,
    Clamp,
    Mix,
    Step,
    SmoothStep,
    Length,
    Dot,
    Cross,
    Norm
}

#[derive(Debug, Clone)]
pub enum AstNode {
    V(Val),
    Assign(Spur, Box<Ast>),
    Block(Vec<Ast>),
    VecLiteral(Box<Ast>, Box<Ast>, Option<Box<Ast>>, Option<Box<Ast>>),
    VecRepeated(Box<Ast>, Box<Ast>),
    VecAccess(Box<Ast>, Swizzle),
    Ident(Spur),
    Return(Box<Ast>),
    Give(Box<Ast>),
    BinOp(Box<Ast>, Op, Box<Ast>),
    If {
        cond: Box<Ast>,
        true_ret: Box<Ast>,
        false_ret: Box<Ast>,
    },
    Call(BuiltIn, Vec<Ast>),
}

#[derive(Debug, Clone)]
pub struct Ast {
    pub node: AstNode,
    pub line: usize,
}

impl Ast {
    pub fn new(node: AstNode, line: usize) -> Ast {
        Ast { node, line }
    }
}

#[derive(Default)]
pub struct Env {
    vars: HashMap<Spur, Val>,
    pub ret: Option<Val>,
    parent: Option<Box<Env>>,
}

impl Env {
    fn child(self) -> Self {
        Self {
            vars: Default::default(),
            parent: Some(Box::new(self)),
            ret: None,
        }
    }

    fn get(&self, ident: Spur) -> Option<Val> {
        self.vars
            .get(&ident)
            .map(|x| *x) // .as_deref()
            .or_else(|| self.parent.as_ref().and_then(|p| p.get(ident)))
    }

    pub fn set(&mut self, ident: Spur, to: Val) {
        if self.vars.contains_key(&ident) || self.get(ident).is_none() {
            self.vars.insert(ident, to);
        } else if let Some(p) = self.parent.as_mut() {
            p.set(ident, to);
        }
    }

    fn return_val(&mut self, val: Val) {
        self.ret = Some(val);
        if let Some(p) = self.parent.as_mut() {
            p.return_val(val)
        }
    }
}

pub struct EvalRet {
    pub env: Env,
    val: Option<Val>,
    give: Option<Val>,
}

struct ERVal {
    env: Env,
    val: Val,
}

impl EvalRet {
    fn new(env: Env) -> Self {
        Self {
            env,
            val: None,
            give: None,
        }
    }

    fn with_val(self, val: Option<Val>) -> Self {
        Self { val, ..self }
    }

    fn with_give(self, give: Val) -> Self {
        Self {
            give: Some(give),
            ..self
        }
    }

    fn needs_val(self) -> ERVal {
        ERVal {
            env: self.env,
            val: need_val(self.val),
        }
    }
}

fn need_val(v: Option<Val>) -> Val {
    v.expect("context requires associated expression to return a value")
}

pub fn eval(ast: &Ast, e: Env) -> EvalRet {
    use AstNode::*;

    match &ast.node {
        &V(v) => EvalRet::new(e).with_val(Some(v)),
        Assign(i, to) => {
            let ERVal { mut env, val } = eval(&to, e).needs_val();
            env.set(*i, val);
            EvalRet::new(env).with_val(Some(val))
        }
        Block(nodes) => {
            let EvalRet { env, give, .. } =
                nodes.iter().fold(EvalRet::new(e.child()), |acc, node| {
                    if acc.give.or(acc.env.ret).is_some() {
                        acc
                    } else {
                        eval(node, acc.env)
                    }
                });
            EvalRet::new(*env.parent.unwrap()).with_val(env.ret.or(give))
        }
        VecRepeated(value, length) => {
            let ERVal { val: value, env } = eval(value, e).needs_val();
            let ERVal { val: length, env } = eval(length, env).needs_val();

            match (value, length) {
                (Float(v), Float(l)) => {
                    match l as usize {
                        2 => EvalRet::new(env).with_val(Some(Val::Vec2(v, v))),
                        3 => EvalRet::new(env).with_val(Some(Val::Vec3(v, v, v))),
                        4 => EvalRet::new(env).with_val(Some(Val::Vec4(v, v, v, v))),
                        n => {
                            panic!("{} is an invalid vector length; must be 2 <= x <= 4", n)
                        }
                    }
                }
                _ => report(ErrorType::Runtime, ast.line, "Both X and L in [X; L] in vector literal must evaluate to scalars"),
            }
        }
        VecLiteral(xast, yast, None, None) => {
            let ERVal { val: xval, env } = eval(&xast, e).needs_val();
            let ERVal { val: yval, env } = eval(&yast, env).needs_val();
            match (xval, yval) {
                (Float(x), Float(y)) => EvalRet::new(env).with_val(Some(Val::Vec2(x, y))),
                _ => panic!(),
            }
        }
        VecLiteral(_, _, None, Some(_)) => panic!("fuck you"),
        VecLiteral(xast, yast, Some(zast), None) => {
            let ERVal { val: xval, env } = eval(&xast, e).needs_val();
            let ERVal { val: yval, env } = eval(&yast, env).needs_val();
            let ERVal { val: zval, env } = eval(&zast, env).needs_val();
            match (xval, yval, zval) {
                (Float(x), Float(y), Float(z)) => {
                    EvalRet::new(env).with_val(Some(Val::Vec3(x, y, z)))
                }
                _ => panic!(),
            }
        }
        VecLiteral(xast, yast, Some(zast), Some(wast)) => {
            let ERVal { val: xval, env } = eval(&xast, e).needs_val();
            let ERVal { val: yval, env } = eval(&yast, env).needs_val();
            let ERVal { val: zval, env } = eval(&zast, env).needs_val();
            let ERVal { val: wval, env } = eval(&wast, env).needs_val();
            match (xval, yval, zval, wval) {
                (Float(x), Float(y), Float(z), Float(w)) => {
                    EvalRet::new(env).with_val(Some(Val::Vec4(x, y, z, w)))
                }
                _ => panic!(),
            }
        }
        VecAccess(access_me, swiz) => {
            let ERVal { env, val } = eval(&access_me, e).needs_val();
            if let Val::Float(x) = val {
                panic!("Expected vec, found {}", x)
            }
            EvalRet::new(env).with_val(Some(match *swiz {
                Swizzle(x, None, None, None) => Val::Float(val.get_field(x)),
                Swizzle(x, Some(y), None, None) => Val::Vec2(val.get_field(x), val.get_field(y)),
                Swizzle(x, Some(y), Some(z), None) => {
                    Val::Vec3(val.get_field(x), val.get_field(y), val.get_field(z))
                }
                Swizzle(x, Some(y), Some(z), Some(w)) => Val::Vec4(
                    val.get_field(x),
                    val.get_field(y),
                    val.get_field(z),
                    val.get_field(w),
                ),
                _ => report(ErrorType::Runtime, ast.line, "Invalid swizzle")
            }))
        }
        &Ident(i) => {
            let val = e.get(i);
            EvalRet::new(e).with_val(val)
        }
        Return(v) => {
            let ERVal { mut env, val } = eval(&v, e).needs_val();
            env.return_val(val);
            EvalRet::new(env)
        }
        Give(g) => {
            let ERVal { env, val } = eval(&g, e).needs_val();
            EvalRet::new(env).with_give(val)
        }
        BinOp(lhs, op, rhs) => {
            let ERVal { env, val: lval } = eval(&lhs, e).needs_val();
            let ERVal { env, val: rval } = eval(&rhs, env).needs_val();

            use Val::*;
            EvalRet::new(env).with_val(Some(match (lval, op, rval) {
                (Float(_), Op::Sub, Vec2(_, _) | Vec3(_, _, _) | Vec4(_, _, _, _))
                | (Float(_), Op::Add, Vec2(_, _) | Vec3(_, _, _) | Vec4(_, _, _, _))
                | (Float(_), Op::Mul, Vec2(_, _) | Vec3(_, _, _) | Vec4(_, _, _, _))
                | (Float(_), Op::Div, Vec2(_, _) | Vec3(_, _, _) | Vec4(_, _, _, _)) => {
                    // TODO: check this in static analysis pass
                    report(ErrorType::Runtime, ast.line, "Unexpected float on lhs and vector on rhs")
                }
                (_, Op::Sub, _) => lval.zipmap(rval, |l, r| l - r),
                (_, Op::Add, _) => lval.zipmap(rval, |l, r| l + r),
                (_, Op::Mul, _) => lval.zipmap(rval, |l, r| l * r),
                (_, Op::Div, _) => lval.zipmap(rval, |l, r| l / r),
                (Float(l), Op::More, Float(r)) => Float((l > r) as i32 as f32),
                (Float(l), Op::Less, Float(r)) => Float((l < r) as i32 as f32),
                (Float(l), Op::MoreEq, Float(r)) => Float((l <= r) as i32 as f32),
                (Float(l), Op::LessEq, Float(r)) => Float((l >= r) as i32 as f32),
                _ => report(
                    ErrorType::Runtime,
                    ast.line,
                    format!("Unexpected scalar/vector binary operation relationship, got {:#?} {:#?} {:#?}", lval, op, rval).as_str(),
                )
            }))
        }
        If {
            cond,
            true_ret,
            false_ret,
        } => {
            let ERVal { env, val: condval } = eval(&cond, e).needs_val();
            match condval {
                Val::Float(f) if f == 1.0 => eval(&true_ret, env),
                Val::Float(_) => eval(&false_ret, env),
                _ => report(ErrorType::Runtime, ast.line, format!("Expected scalar conditional in if expression, got {:#?}", condval).as_str()),
            }
        }
        Call(builtin, args) => {
            let len = args.len();
            let (env, mut vals) =
                args.iter()
                    .fold((e, Vec::with_capacity(len)), |(env, mut vals), arg| {
                        let ERVal { env, val } = eval(arg, env).needs_val();
                        vals.push(val);
                        (env, vals)
                    });

            match *builtin {
                BuiltIn::Dist => {
                    match (len, vals.pop(), vals.pop()) {
                        (2, Some(Val::Float(x0)), Some(Val::Float(x1))) => {
                            EvalRet::new(env).with_val(Some(Val::Float((x1 - x0).abs())))
                        }
                        (2, Some(Val::Vec2(x0, y0)), Some(Val::Vec2(x1, y1))) => {
                            EvalRet::new(env).with_val(Some(Val::Float(((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt())))
                        } 
                        (2, Some(Val::Vec3(x0, y0, z0)), Some(Val::Vec3(x1, y1, z1))) => {
                            EvalRet::new(env).with_val(Some(Val::Float(((x1 - x0).powi(2) + (y1 - y0).powi(2) + (z1 - z0).powi(2)).sqrt())))
                        }
                        (2, Some(Val::Vec4(x0, y0, z0, w0)), Some(Val::Vec4(x1, y1, z1, w1))) => {
                            EvalRet::new(env).with_val(Some(Val::Float(((x1 - x0).powi(2) + (y1 - y0).powi(2) + (z1 - z0).powi(2) + (w1 - w0).powi(2)).sqrt())))
                        }
                        (2, v1, v2) => report(ErrorType::Runtime, ast.line, format!("Unexpected inputs to dist(a, b), got {:#?} and {:#?}", v1, v2).as_str()),
                        (n, _, _) => report(ErrorType::Runtime, ast.line, format!("Expected 2 inputs to dist(a, b), got {}", n).as_str())
                    }
                }
                BuiltIn::Radians => {
                    let radians = |deg: f32| deg * std::f32::consts::PI / 180.0;

                    match (len, vals.pop()) {
                        (1, Some(Val::Float(x))) => {
                            EvalRet::new(env).with_val(Some(Val::Float(radians(x))))
                        }
                        (1, Some(Val::Vec2(x, y))) => {
                            EvalRet::new(env).with_val(Some(Val::Vec2(radians(x), radians(y))))
                        }
                        (1, Some(Val::Vec3(x, y, z))) => {
                            EvalRet::new(env).with_val(Some(Val::Vec3(radians(x), radians(y), radians(z))))
                        }
                        (1, Some(Val::Vec4(x, y, z, w))) => {
                            EvalRet::new(env).with_val(Some(Val::Vec4(radians(x), radians(y), radians(z), radians(w))))
                        }
                        (n, _) => report(ErrorType::Runtime, ast.line, format!("Expected 1 input to radians(a), got {}", n).as_str()),
                        _=> report(ErrorType::Runtime, ast.line, "Unexpected input to radians(a), expected float, vec2, vec3, vec4")
                    }
                },
                BuiltIn::Degrees => {
                    let degrees = |rad: f32| rad * 180.0 / std::f32::consts::PI;

                    match (len, vals.pop()) {
                        (1, Some(Val::Float(x))) => {
                            EvalRet::new(env).with_val(Some(Val::Float(degrees(x))))
                        }
                        (1, Some(Val::Vec2(x, y))) => {
                            EvalRet::new(env).with_val(Some(Val::Vec2(degrees(x), degrees(y))))
                        }
                        (1, Some(Val::Vec3(x, y, z))) => {
                            EvalRet::new(env).with_val(Some(Val::Vec3(degrees(x), degrees(y), degrees(z))))
                        }
                        (1, Some(Val::Vec4(x, y, z, w))) => {
                            EvalRet::new(env).with_val(Some(Val::Vec4(degrees(x), degrees(y), degrees(z), degrees(w))))
                        }
                        _ => report(ErrorType::Runtime, ast.line, "Unexpected inputs to radians(), expected float, vec2, vec3, vec4")
                    }
                },
                BuiltIn::Pow => {
                    match (len, vals.pop(), vals.pop()) {
                        (2, Some(Val::Float(x)), Some(Val::Float(exp))) => {
                            EvalRet::new(env).with_val(Some(Val::Float(x.powf(exp))))
                        }
                        (2, Some(Val::Vec2(x, y)), Some(Val::Float(exp))) => {
                            EvalRet::new(env).with_val(Some(Val::Vec2(x.powf(exp), y.powf(exp))))
                        }
                        (2, Some(Val::Vec3(x, y, z)), Some(Val::Float(exp))) => {
                            EvalRet::new(env).with_val(Some(Val::Vec3(x.powf(exp), y.powf(exp), z.powf(exp))))
                        }
                        (2, Some(Val::Vec4(x, y, z, w)), Some(Val::Float(exp))) => {
                            EvalRet::new(env).with_val(Some(Val::Vec4(x.powf(exp), y.powf(exp), z.powf(exp), w.powf(exp))))
                        }
                        (2, x, exp) => report(ErrorType::Runtime, ast.line, format!("Expected pow(vec*, float), got {:?}, {:?}", x, exp).as_str()),
                        (n, _, _) => report(ErrorType::Runtime, ast.line, format!("Expected two inputs to pow(), got {}", n).as_str())
                    }
                },
                BuiltIn::Sin => todo!(),
                BuiltIn::Cos => todo!(),
                BuiltIn::Tan => todo!(),
                BuiltIn::Asin => todo!(),
                BuiltIn::Acos => todo!(),
                BuiltIn::Atan => todo!(),
                BuiltIn::Exp => todo!(),
                BuiltIn::Log => todo!(),
                BuiltIn::Sqrt => todo!(),
                BuiltIn::InverseSqrt => todo!(),
                BuiltIn::Abs => todo!(),
                BuiltIn::Sign => todo!(),
                BuiltIn::Floor => todo!(),
                BuiltIn::Ceil => todo!(),
                BuiltIn::Fract => todo!(),
                BuiltIn::Mod => todo!(),
                BuiltIn::Min => todo!(),
                BuiltIn::Max => todo!(),
                BuiltIn::Clamp => todo!(),
                BuiltIn::Mix => todo!(),
                BuiltIn::Step => todo!(),
                BuiltIn::SmoothStep => todo!(),
                BuiltIn::Length => todo!(),
                BuiltIn::Dot => todo!(),
                BuiltIn::Cross => todo!(),
                BuiltIn::Norm => todo!(),
            }
        }
    }
}
