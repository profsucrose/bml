use std::collections::HashMap;

pub mod examples;

#[derive(Debug, Copy, Clone)]
pub enum Op {
  Add, Sub,
  Div, Mul,
  More, Less,
  MoreEq, LessEq,
  Equal
}

#[derive(Copy, Clone, Debug)]
pub enum Field { X, Y, Z, W }
#[derive(Copy, Clone, Debug)]
pub struct Swizzle(Field, Option<Field>, Option<Field>, Option<Field>);

#[derive(PartialEq, Clone, Copy, Debug)]
pub enum Val {
  Float(f32),
  Vec2(f32, f32),
  Vec3(f32, f32, f32),
  Vec4(f32, f32, f32, f32),
}
impl Val {
  pub fn map<F: FnMut(f32) -> f32>(self, mut f: F) -> Val {
    use Val::*;
    match self {
      Float(x) => Float(f(x)),
      Vec2(x, y) => Vec2(f(x), f(y)),
      Vec3(x, y, z) => Vec3(f(x), f(y), f(z)),
      Vec4(x, y, z, w) => Vec4(f(x), f(y), f(z), f(w)),
    }
  }

  pub fn zipmap<F: FnMut(f32, f32) -> f32>(self, o: Self, mut f: F) -> Val {
    use Val::*;
    match (self, o) {
      (Float(l), Float(r)) => Float(f(l, r)),
      (Vec2(lx, ly), Float(r)) => Vec2(f(lx, r), f(ly, r)),
      (Vec3(lx, ly, lz), Float(r)) => Vec3(f(lx, r), f(ly, r), f(lz, r)),
      (Vec4(lx, ly, lz, lw), Float(r)) =>
        Vec4(f(lx, r), f(ly, r), f(lz, r), f(lw, r)),

      (Vec2(lx, ly), Vec2(rx, ry)) => Vec2(f(lx, rx), f(ly, ry)),
      (Vec3(lx, ly, lz), Vec3(rx, ry, rz)) =>
        Vec3(f(lx, rx), f(ly, ry), f(lz, rz)),
      (Vec4(lx, ly, lz, lw), Vec4(rx, ry, rz, rw)) =>
        Vec4(f(lx, rx), f(ly, ry), f(lz, rz), f(lw, rw)),
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
      }
      Field::Y => match *self {
        Float(_) => panic!("Float has no y field"),
        Vec2(_, y) => y,
        Vec3(_, y, _) => y,
        Vec4(_, y, _, _) => y,
      }
      Field::Z => match *self {
        Float(_) => panic!("Float has no z field"),
        Vec2(_, _) => panic!("Vec2 has no z field"),
        Vec3(_, _, z) => z,
        Vec4(_, _, z, _) => z,
      }
      Field::W => match *self {
        Float(_) => panic!("Float has no w field"),
        Vec2(_, _) => panic!("Vec2 has no w field"),
        Vec3(_, _, _) => panic!("Vec3 has no w field"),
        Vec4(_, _, _, w) => w,
      }
    }
  }
}

#[derive(Debug)]
pub enum BuiltIn { Dist }

#[derive(Debug)]
pub enum Ast {
  V(Val),
  Assign(i32, Box<Ast>), /* i32 will prolly become lasso::Spur */
  Block(Vec<Ast>),
  VecLiteral(Box<Ast>, Box<Ast>, Option<Box<Ast>>, Option<Box<Ast>>),
  VecAccess(Box<Ast>, Swizzle),
  Ident(i32), /* will prolly become lasso::Spur */
  Return(Box<Ast>),
  Give(Box<Ast>),
  BinOp(Box<Ast>, Op, Box<Ast>),
  If {
    cond: Box<Ast>,
    true_ret: Box<Ast>,
    false_ret: Box<Ast>
  },
  Call(BuiltIn, Vec<Ast>),
}

#[test]
fn invert_color_ast() {
  let (env, ast) = examples::invert_color();
  env.set(4, Val::Vec4(1.0, 0.0, 0.5, 0.0));
  assert_eq!(
    Some(Val::Vec4(0.0, 1.0, 0.5, 1.0)),
    eval(ast, env).env.ret
  );
}

#[test]
fn closing_circle_ast() {
  let (env, ast) = examples::closing_circle();
  env.set(2, Val::Vec2(100.0, 100.0)); /* resolution */
  env.set(3, Val::Vec2( 90.0,  90.0)); /* coord */
  env.set(4, Val::Vec4(1.0, 1.0, 1.0, 1.0)); /* frag */
  env.set(5, Val::Float(10.0)); /* max_frame */
  env.set(6, Val::Float( 3.0)); /* frame */

  assert_eq!(
    Some(Val::Vec4(1.0, 1.0, 1.0, 1.0)),
    eval(ast, env).env.ret
  );
}

#[derive(Default, Debug)]
pub struct Env {
  vars: HashMap<i32, Val>,
  pub ret: Option<Val>,
  parent: Option<Box<Env>>,
}

impl Env {
  fn child(self) -> Self {
    Self {
      vars: Default::default(),
      parent: Some(Box::new(self)),
      ret: None
    }
  }
  pub fn get(&self, ident: i32) -> Option<Val> {
    self.vars.get(&ident)
      .map(|x| *x) // .as_deref()
      .or_else(|| self.parent.as_ref().and_then(|p| p.get(ident)))
  }
  pub fn set(&mut self, ident: i32, to: Val) {
    if self.vars.contains_key(&ident) || self.get(ident).is_none() {
      self.vars.insert(ident, to);
    } else if let Some(p) = self.parent.as_mut() {
      p.set(ident, to);
    }
  }
  fn return_val(&mut self, val: Val) {
    self.ret = Some(val);
    if let Some(p) = self.parent.as_mut() { p.return_val(val) }
  }
}

pub struct EvalRet {
  pub env: Env,
  val: Option<Val>,
  give: Option<Val>,
}
struct ERVal { env: Env, val: Val }
impl EvalRet {
  fn new(env: Env) -> Self { Self { env, val: None, give: None } }
  fn with_val(self, val: Option<Val>) -> Self { Self { val, ..self } }
  fn with_give(self, give: Val) -> Self { Self { give: Some(give), ..self }}
  fn needs_val(self) -> ERVal {
    ERVal { env: self.env, val: need_val(self.val) }
  }
}

fn need_val(v: Option<Val>) -> Val {
  v.expect("context requires associated expression to return a value")
}

pub fn eval(ast: &Ast, e: Env) -> EvalRet {
  use Ast::*;
  use Val::Float;

  match ast {
    &V(v) => EvalRet::new(e).with_val(Some(v)),
    Assign(i, to) => {
      let ERVal { mut env, val } = eval(&to, e).needs_val();
      env.set(*i, val);
      EvalRet::new(env)
    },
    Block(nodes) => {
      let EvalRet { env, give, .. } = nodes.iter().fold(
        EvalRet::new(e.child()),
        |acc, node| {
          if acc.give.or(acc.env.ret).is_some() {
            acc
          } else {
            eval(node, acc.env)
          }
        },
      );
      EvalRet::new(*env.parent.unwrap()).with_val(env.ret.or(give))
    }
    VecLiteral(xast, yast, None, None) => {
      let ERVal { val: xval, env } = eval(&xast, e).needs_val();
      let ERVal { val: yval, env } = eval(&yast, env).needs_val();
      match (xval, yval) {
        (Float(x), Float(y)) =>
          EvalRet::new(env).with_val(Some(Val::Vec2(x, y))),
        _ => panic!()
      }
    }
    VecLiteral(_, _, None, Some(_)) => panic!("fuck you"),
    VecLiteral(xast, yast, Some(zast), None) => {
      let ERVal { val: xval, env } = eval(&xast, e).needs_val();
      let ERVal { val: yval, env } = eval(&yast, env).needs_val();
      let ERVal { val: zval, env } = eval(&zast, env).needs_val();
      match (xval, yval, zval) {
        (Float(x), Float(y), Float(z)) =>
          EvalRet::new(env).with_val(Some(Val::Vec3(x, y, z))),
        _ => panic!()
      }
    }
    VecLiteral(xast, yast, Some(zast), Some(wast)) => {
      let ERVal { val: xval, env } = eval(&xast, e).needs_val();
      let ERVal { val: yval, env } = eval(&yast, env).needs_val();
      let ERVal { val: zval, env } = eval(&zast, env).needs_val();
      let ERVal { val: wval, env } = eval(&wast, env).needs_val();
      match (xval, yval, zval, wval) {
        (Float(x), Float(y), Float(z), Float(w)) => 
          EvalRet::new(env).with_val(Some(Val::Vec4(x, y, z, w))),
        _ => panic!()
      }
    }
    VecAccess(access_me, swiz) => {
      let ERVal { env, val } = eval(&access_me, e).needs_val();
      if let Val::Float(x) = val { panic!("Expected vec, found {}", x) }
      EvalRet::new(env).with_val(Some(match *swiz {
        Swizzle(x, None   , None   , None   ) =>
          Val::Float(val.get_field(x)),
        Swizzle(x, Some(y), None   , None   ) =>
          Val::Vec2(val.get_field(x), val.get_field(y)),
        Swizzle(x, Some(y), Some(z), None   ) =>
          Val::Vec3(val.get_field(x), val.get_field(y), val.get_field(z)),
        Swizzle(x, Some(y), Some(z), Some(w)) =>
          Val::Vec4(val.get_field(x), val.get_field(y),
                    val.get_field(z), val.get_field(w)),
        _ => panic!("invalid swizzle")
      }))
    },
    &Ident(i) => {
      let val = e.get(i).unwrap_or_else(|| panic!("no such ident, {}", i));
      EvalRet::new(e).with_val(Some(val))
    }
    Return(v) => {
      let ERVal { mut env, val } = eval(&v, e).needs_val();
      env.return_val(val);
      EvalRet::new(env)
    },
    Give(g) => {
      let ERVal { env, val } = eval(&g, e).needs_val();
      EvalRet::new(env).with_give(val)
    },
    BinOp(lhs, op, rhs) => {
      let ERVal { env, val: lval } = eval(&lhs, e).needs_val();
      let ERVal { env, val: rval } = eval(&rhs, env).needs_val();

      use Val::*;
      EvalRet::new(env).with_val(Some(match (lval, op, rval) {
        (Float(_), Op::Sub, Vec2(_,_) | Vec3(_,_,_) | Vec4(_,_,_,_)) |
          (Float(_), Op::Add, Vec2(_,_) | Vec3(_,_,_) | Vec4(_,_,_,_)) |
          (Float(_), Op::Mul, Vec2(_,_) | Vec3(_,_,_) | Vec4(_,_,_,_)) |
          (Float(_), Op::Div, Vec2(_,_) | Vec3(_,_,_) | Vec4(_,_,_,_)) => {
            panic!("you cannot have a float lhs and a vec rhs \
            (TODO: static analysis this)");
        }
        (_, Op::Sub, _) => lval.zipmap(rval, |l, r| l - r),
        (_, Op::Add, _) => lval.zipmap(rval, |l, r| l + r),
        (_, Op::Mul, _) => lval.zipmap(rval, |l, r| l * r),
        (_, Op::Div, _) => lval.zipmap(rval, |l, r| l / r),
        // (Float(l), Op::More, Float(r)) => { dbg!(l > r); Float((l > r) as i32 as f32)},
        (Float(l), Op::More, Float(r)) => Float((l > r) as i32 as f32),
        (Float(l), Op::Less, Float(r)) => Float((l < r) as i32 as f32),
        (Float(l), Op::MoreEq, Float(r)) => Float((l <= r) as i32 as f32),
        (Float(l), Op::LessEq, Float(r)) => Float((l >= r) as i32 as f32),
        _ => panic!(
          "unsupported scalar/vector binary operation relationship \
          {:#?} {:#?} {:#?}",
          lval, op, rval
        )
      }))
    },
    If { cond, true_ret, false_ret } => {
      let ERVal { env, val: condval } = eval(&cond, e).needs_val();
      match condval {
        Val::Float(f) if f == 1.0 => eval(&true_ret, env),
        Val::Float(_) => eval(&false_ret, env),
        _ => panic!("If logic expects scalar conditionals"),
      }
    }
    Call(BuiltIn::Dist, args) => {
      let len = args.len();
      let (env, mut vals) = args.iter().fold(
        (e, Vec::with_capacity(len)),
        |(env, mut vals), arg| {
          let ERVal { env, val } = eval(arg, env).needs_val();
          vals.push(val);
          (env, vals)
        },
      );
      match (len, vals.pop(), vals.pop()) {
        (2, Some(Val::Vec2(x0, y0)), Some(Val::Vec2(x1, y1))) => {
          let (dx, dy) = (x0 - x1, y0 - y1);
          EvalRet::new(env).with_val(Some(Val::Float((dx*dx + dy*dy).sqrt())))
        }
        _ => panic!("unexpected inputs to dist (tbf it's janked rn)"),
      }
    }
  }
}
