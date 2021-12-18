pub enum Op {
  Add, Sub,
  Div, Mul,
  More, Less,
  MoreEq, LessEq,
  Equal
}

#[derive(Debug, PartialEq)]
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
}

pub enum Field { X, Y, Z, W }
struct Swizzle(Field, Option<Field>, Option<Field>, Option<Field>);

enum BuiltIn { Sqrt, Dist }

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
  use Ast::*;
  fn b(a: Ast) -> Box<Ast> { Box::new(a) }
  Return(b(VecLiteral(
    b(BinOp(
      b(V(Val::Float(1.0))),
      Op::Sub,
      b(VecAccess(b(Ident(0)), Swizzle(Field::X, None, None, None))),
    )),
    b(BinOp(
      b(V(Val::Float(1.0))),
      Op::Sub,
      b(VecAccess(b(Ident(0)), Swizzle(Field::Y, None, None, None))),
    )),
    Some(b(BinOp(
      b(V(Val::Float(1.0))),
      Op::Sub,
      b(VecAccess(b(Ident(0)), Swizzle(Field::Z, None, None, None))),
    ))),
    Some(b(V(Val::Float(1.0)))),
  )));
}

#[test]
fn closing_circle_ast() {
  use Ast::*;
  fn b(a: Ast) -> Box<Ast> { Box::new(a) }
  Block(vec![
    Assign(0, b(BinOp(b(Ident(2)), Op::Div, b(Ident(3))))),
    Return(b(If {
      cond: b(Call(BuiltIn::Dist, vec![Ident(0), V(Val::Vec2(0.5, 0.5))])),
      true_ret: b(Ident(2)),
      false_ret: b(V(Val::Vec4(0.0, 0.0, 0.0, 0.0))),
    }))
  ]);
}
