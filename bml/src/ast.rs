use core::panic;
use image::{ImageBuffer, Rgba};
use lasso::{Rodeo, Spur};
use std::collections::HashMap;

use crate::logger::{report, ErrorType};

#[derive(Debug, Copy, Clone, PartialEq)]
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

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Field {
    X,
    Y,
    Z,
    W,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Swizzle(
    pub Field,
    pub Option<Field>,
    pub Option<Field>,
    pub Option<Field>,
);

// #[derive(PartialEq, Clone, Copy, Debug)]
// pub enum Vector {
//     Vec2(f32, f32),
//     Vec3(f32, f32, f32),
//     Vec4(f32, f32, f32, f32),
// }

// #[derive(PartialEq, Clone, Copy, Debug)]
// pub enum Matrix {
//     Mat2([Vector; 2]),
//     Mat3([Vector; 3]),
//     Mat4([Vector; 4]),
// }

// #[derive(PartialEq, Clone, Copy, Debug)]
// pub enum Val {
//     Float(f32),
//     Vec(Vector),
//     Mat(Matrix)
// }

#[derive(PartialEq, Clone, Copy, Debug)]
pub enum Val {
    Float(f32),
    Vec2(f32, f32),
    Vec3(f32, f32, f32),
    Vec4(f32, f32, f32, f32),

    // matrices are column-major
    Mat2([f32; 2], [f32; 2]),
    Mat3([f32; 3], [f32; 3], [f32; 3]),
    Mat4([f32; 4], [f32; 4], [f32; 4], [f32; 4])
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
    pub fn float(self) -> Option<f32> {
        match self {
            Self::Float(x) => Some(x),
            _ => None,
        }
    }

    pub fn map<F: FnMut(f32) -> f32>(self, mut f: F) -> Val {
        match self {
            Float(x) => Float(f(x)),
            Vec2(x, y) => Vec2(f(x), f(y)),
            Vec3(x, y, z) => Vec3(f(x), f(y), f(z)),
            Vec4(x, y, z, w) => Vec4(f(x), f(y), f(z), f(w)),
            Mat2(row0, row1) => Mat2(row0.map(|x| f(x)), row1.map(|x| f(x))),
            Mat3(row0, row1, row2) => {
                Mat3(row0.map(|x| f(x)), row1.map(|x| f(x)), row2.map(|x| f(x)))
            }
            Mat4(row0, row1, row2, row3) => Mat4(
                row0.map(|x| f(x)),
                row1.map(|x| f(x)),
                row2.map(|x| f(x)),
                row3.map(|x| f(x)),
            ),
        }
    }

    pub fn zipmap3<F: FnMut(f32, f32, f32) -> f32>(self, o1: Self, o2: Self, mut f: F) -> Val {
        match (self, o1, o2) {
            (Float(x), Float(y), Float(z)) => Float(f(x, y, z)),
            (Vec2(x0, x1), Vec2(y0, y1), Vec2(z0, z1)) => Vec2(f(x0, y0, z0), f(x1, y1, z1)),
            (Vec3(x0, x1, x2), Vec3(y0, y1, y2), Vec3(z0, z1, z2)) => {
                Vec3(f(x0, y0, z0), f(x1, y1, z1), f(x2, y2, z2))
            }
            (Vec4(x0, x1, x2, x3), Vec4(y0, y1, y2, y3), Vec4(z0, z1, z2, z3)) => {
                Vec4(f(x0, y0, z0), f(x1, y1, z1), f(x2, y2, z2), f(x3, y3, z3))
            }
            _ => panic!(
                "{}",
                format!(
                    "expected all args to zipmap3 be vectors of the same type, got {:?}, {:?}, {:?}",
                    self, o1, o2
                )
            ),
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

            _ => panic!("Unexpected relationship between scalars, vectors or matrices in operation")
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
                _ => panic!("Tried to swizzle matrix"),
            },
            Field::Y => match *self {
                Float(_) => panic!("Float has no y field"),
                Vec2(_, y) => y,
                Vec3(_, y, _) => y,
                Vec4(_, y, _, _) => y,
                _ => panic!("Tried to swizzle matrix"),
            },
            Field::Z => match *self {
                Float(_) => panic!("Float has no z field"),
                Vec2(_, _) => panic!("Vec2 has no z field"),
                Vec3(_, _, z) => z,
                Vec4(_, _, z, _) => z,
                _ => panic!("Tried to swizzle matrix"),
            },
            Field::W => match *self {
                Float(_) => panic!("Float has no w field"),
                Vec2(_, _) => panic!("Vec2 has no w field"),
                Vec3(_, _, _) => panic!("Vec3 has no w field"),
                Vec4(_, _, _, w) => w,
                _ => panic!("Tried to swizzle matrix"),
            },
        }
    }

    pub fn translate(x: f32, y: f32, z: f32) -> Val {
        Mat4(
            [x,   0.0, 0.0, 0.0],
            [0.0,   y, 0.0, 0.0],
            [0.0, 0.0,   z, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        )
    }

    pub fn rotate_x(angle: f32) -> Val {
        let c = angle.cos();
        let s = angle.sin();

        Mat4(
            [1.0, 0.0, 0.0, 0.0],
            [0.0,   c,   s, 0.0],
            [0.0,  -s,   c, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        )
    }

    pub fn rotate_y(angle: f32) -> Val {
        let c = angle.cos();
        let s = angle.sin();

        Mat4(
            [  c, 0.0,  -s, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [  s, 0.0,   c, 0.0],
            [  s, 0.0,   c, 1.0]
        )
    }

    pub fn rotate_z(angle: f32) -> Val {
        let c = angle.cos();
        let s = angle.sin();

        Mat4(
            [  c,   s, 0.0, 0.0],
            [ -s,   c, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        )
    }

    pub fn rotate(yaw: f32, pitch: f32, roll: f32) -> Val {
        Val::rotate_z(yaw).mult(Val::rotate_y(pitch).mult(Val::rotate_x(roll)).unwrap()).unwrap()
    }

    pub fn scale(s: f32) -> Val {
        Mat4(
            [  s, 0.0, 0.0, 0.0],
            [0.0,   s, 0.0, 0.0],
            [0.0, 0.0,   s, 0.0],
            [0.0, 0.0, 0.0,   s]
        )
    }

    pub fn ortho(near: f32, far: f32, left: f32, right: f32, top: f32, bottom: f32) -> Val {
        Mat4(
            [ 2.0 / (right - left),                 0.0,                0.0, -(right + left) / (right - left) ],
            [                  0.0, 2.0 / (top - bottom),               0.0, -(top + bottom) / (top - bottom) ],
            [                  0.0,                  0.0, -2.0/(far - near),     -(far + near) / (far - near) ],
            [                  0.0,                  0.0,               0.0,                             1.0 ]
        )
    }

    pub fn perspective(fov: f32, near: f32, far: f32) -> Val {
        let s = 1.0 / (fov / 2.0 * std::f32::consts::PI / 180.0);

        Mat4(
            [s, 0.0, 0.0, 0.0],
            [0.0, s, 0.0, 0.0],
            [0.0, 0.0, -far / (far - near), -1.0],
            [0.0, 0.0, -(far * near) / (far - near), 0.0]
        )
    }

    pub fn lookat(from: Val, to: Val) -> Val {
        let up = Vec3(0.0, 1.0, 0.0);
        let forward = from.zipmap(to, |l, r| l -r).norm();
        let right = up.norm().cross(forward).unwrap();

        // center row is up-axis Vec3(0, 1, 0)

        Mat4(
            [right.get_field(Field::X), 0.0, forward.get_field(Field::X), from.get_field(Field::X)],
            [right.get_field(Field::Y), 1.0, forward.get_field(Field::Y), from.get_field(Field::Y)],
            [right.get_field(Field::Z), 0.0, forward.get_field(Field::Z), from.get_field(Field::Z)],
            [                      0.0, 0.0,                         0.0,                      1.0]
        )
    }

    pub fn mult(&self, o: Self) -> Result<Val, String> {
        match (*self, o) {
            (Float(_), Float(_))
                | (Vec2(_, _), Vec2(_, _))
                | (Vec3(_, _, _), Vec3(_, _, _))
                | (Vec4(_, _, _, _), Vec4(_, _, _, _))
            => Ok(self.zipmap(o, |x, y| x * y)),

            (Float(a), Vec2(_, _) | Vec3(_, _, _) | Vec4(_, _, _, _))
                | (Vec2(_, _) | Vec3(_, _, _) | Vec4(_, _, _, _), Float(a))
            => Ok(o.map(|x| a * x)),

            (Float(x), Mat2(col0, col1)) => Ok(Mat2([col0[0] * x, col0[1] * x], [col1[0] * x, col1[1] * x])),
            (Float(x), Mat3(col0, col1, col2)) => Ok(Mat3([col0[0] * x, col0[1] * x, col0[2] * x], [col1[0] * x, col1[1] * x, col1[2] * x], [col2[0] * x, col2[1] * x, col2[2] * x])),
            (Float(x), Mat4(col0, col1, col2, col3)) => Ok(Mat4([col0[0] * x, col0[1] * x, col0[2] * x, col0[3] * x], [col1[0] * x, col1[1] * x, col1[2] * x, col1[3] * x], [col2[0] * x, col2[1] * x, col2[2] * x, col2[3] * x], [col3[0] * x, col3[1] * x, col3[2] * x, col3[3] * x])),
            
            (Vec2(x, y), Mat2(col0, col1)) => Ok(Vec2(x * col0[0] + y * col0[0], x * col1[0] + y * col1[0])),
            (Vec3(x, y, z), Mat3(col0, col1, col2)) => Ok(Vec3(x * col0[0] + y * col0[1] + z * col0[2], x * col1[0] + y * col1[1] + z * col1[2], x * col2[0] + y * col2[1] + z * col2[2])), 
            (Vec4(x, y, z, w), Mat4(col0, col1, col2, col3)) => Ok(Vec4(x * col0[0] + y * col0[1] + z * col0[2] + w * col0[3], x * col1[0] + y * col1[1] + z * col1[2] + w * col1[3], x * col2[0] + y * col2[1] + z * col2[2] + w * col2[3], x * col3[0] + y * col3[1] + z * col3[2] + w * col3[3])),
            
            (Mat2(col0, col1), Vec2(x, y)) => Ok(Vec2(col0[0] * x + col1[0] * y, col0[1] * x + col1[1] * y)),
            (Mat3(col0, col1, col2), Vec3(x, y, z)) => Ok(Vec3(col0[0] * x + col1[0] * y + col2[0] * z, col0[1] * x + col1[1] * y + col2[1] * z, col0[2] * x + col1[2] * y + col2[2] * z)),
            (Mat4(col0, col1, col2, col3), Vec4(x, y, z, w)) => Ok(Vec4(col0[0] * x + col1[0] * y + col2[0] * z + col3[0] * w, col0[1] * x + col1[1] * y + col2[1] * z + col3[1] * w, col0[2] * x + col1[2] * y + col2[2] * z + col3[2] * w, col0[3] * x + col1[3] * y + col2[3] * z + col3[3] * w)),

            (Mat2(a0, a1), Mat2(b0, b1)) => Ok(
                Mat2(
                    [a0[0] * b0[0] + a1[0] * b0[1], a0[1] * b0[0] + a1[1] * b0[1]],
                    [a0[0] * b1[0] + a1[0] * b1[1], a0[1] * b1[0] + a1[1] * b1[1]]
                )
            ),

            (Mat3(a0, a1, a2), Mat3(b0, b1, b2)) => Ok(
                Mat3(
                    [a0[0] * b0[0] + a1[0] * b0[1] + a2[0] * b0[2], a0[1] * b0[0] + a1[1] * b0[1] + a2[1] * b0[2], a0[2] * b0[0] + a1[2] * b0[1] + a2[2] * b0[2]],
                    [a0[0] * b1[0] + a1[0] * b1[1] + a2[0] * b1[2], a0[1] * b1[0] + a1[1] * b1[1] + a2[1] * b1[2], a0[2] * b1[0] + a1[2] * b1[1] + a2[2] * b1[2]],
                    [a0[0] * b2[0] + a1[0] * b2[1] + a2[0] * b2[2], a0[1] * b2[0] + a1[1] * b2[1] + a2[1] * b2[2], a0[2] * b2[0] + a1[2] * b2[1] + a2[2] * b2[2]]
                )
            ),

            (Mat4(a0, a1, a2, a3), Mat4(b0, b1, b2, b3)) => Ok(
                Mat4(
                    [a0[0] * b0[0] + a1[0] * b0[1] + a2[0] * b0[2] + a3[0] * b0[3], a0[1] * b0[0] + a1[1] * b0[1] + a2[1] * b0[2] + a3[1] * b0[3], a0[2] * b0[0] + a1[2] * b0[1] + a2[2] * b0[2] + a3[2] * b0[3], a0[3] * b0[0] + a1[3] * b0[1] + a2[3] * b0[2] + a3[3] * b0[3]],
                    [a0[0] * b1[0] + a1[0] * b1[1] + a2[0] * b1[2] + a3[0] * b1[3], a0[1] * b1[0] + a1[1] * b1[1] + a2[1] * b1[2] + a3[1] * b1[3], a0[2] * b1[0] + a1[2] * b1[1] + a2[2] * b1[2] + a3[2] * b1[3], a0[3] * b1[0] + a1[3] * b1[1] + a2[3] * b1[2] + a3[3] * b1[3]],
                    [a0[0] * b2[0] + a1[0] * b2[1] + a2[0] * b2[2] + a3[0] * b2[3], a0[1] * b2[0] + a1[1] * b2[1] + a2[1] * b2[2] + a3[1] * b2[3], a0[2] * b2[0] + a1[2] * b2[1] + a2[2] * b2[2] + a3[2] * b2[3], a0[3] * b2[0] + a1[3] * b2[1] + a2[3] * b2[2] + a3[3] * b2[3]],
                    [a0[0] * b3[0] + a1[0] * b3[1] + a2[0] * b3[2] + a3[0] * b3[3], a0[1] * b3[0] + a1[1] * b3[1] + a2[1] * b3[2] + a3[1] * b3[3], a0[2] * b3[0] + a1[2] * b3[1] + a2[2] * b3[2] + a3[2] * b3[3], a0[3] * b3[0] + a1[3] * b3[1] + a2[3] * b3[2] + a3[3] * b3[3]]
                )
            ),

            _ => Err(format!("Expected vecX * vecX, vecX * matX, matX * vecX, float * vecX, or float * matX, got {:?} * {:?}", self, o))
        }
    }

    pub fn index_matrix(&self, f: usize) -> Result<Val, String> {
        match (self, f) {
            (&Mat2(col0, _), 0) => Ok(Vec2(col0[0], col0[1])),
            (&Mat2(_, col1), 1) => Ok(Vec2(col1[0], col1[1])),
            (&Mat2(_, _), index) => Err(format!(
                "Expected 0 <= index <= 1 when accessing mat2, got {}",
                index
            )),

            (&Mat3(col0, _, _), 0) => Ok(Vec3(col0[0], col0[1], col0[2])),
            (&Mat3(_, col1, _), 1) => Ok(Vec3(col1[0], col1[1], col1[2])),
            (&Mat3(_, _, col2), 2) => Ok(Vec3(col2[0], col2[1], col2[2])),
            (&Mat3(_, _, _), index) => Err(format!(
                "Expected 0 <= index <= 2, when accessing mat3, got {}",
                index
            )),

            (&Mat4(col0, _, _, _), 0) => Ok(Vec4(col0[0], col0[1], col0[2], col0[3])),
            (&Mat4(_, col1, _, _), 1) => Ok(Vec4(col1[0], col1[1], col1[2], col1[3])),
            (&Mat4(_, _, col2, _), 2) => Ok(Vec4(col2[0], col2[1], col2[2], col2[3])),
            (&Mat4(_, _, _, col3), 3) => Ok(Vec4(col3[0], col3[1], col3[2], col3[3])),
            (&Mat4(_, _, _, _), index) => Err(format!(
                "Expected 0 <= index <= 3, when accessing mat4, got {}",
                index
            )),

            (accessor, _) => Err(format!("Expected matrix when indexing, got {:?}", accessor)),
        }
    }

    pub fn radians(&self) -> Val {
        self.map(math::rad)
    }

    pub fn degrees(&self) -> Val {
        self.map(math::deg)
    }

    pub fn sin(&self) -> Val {
        self.map(f32::sin)
    }

    pub fn cos(&self) -> Val {
        self.map(f32::cos)
    }

    pub fn tan(&self) -> Val {
        self.map(f32::tan)
    }

    pub fn asin(&self) -> Val {
        self.map(f32::asin)
    }

    pub fn acos(&self) -> Val {
        self.map(f32::acos)
    }

    pub fn atan(&self) -> Val {
        self.map(f32::atan)
    }

    pub fn exp(&self) -> Val {
        self.map(f32::exp)
    }

    pub fn log(&self) -> Val {
        self.map(|x| f32::log(std::f32::consts::E, x))
    }

    pub fn sqrt(&self) -> Val {
        self.map(f32::sqrt)
    }

    pub fn invsqrt(&self) -> Val {
        self.map(math::inv_sqrt)
    }

    pub fn abs(&self) -> Val {
        self.map(f32::abs)
    }

    pub fn sign(&self) -> Val {
        self.map(f32::signum)
    }

    pub fn floor(&self) -> Val {
        self.map(f32::floor)
    }

    pub fn ceil(&self) -> Val {
        self.map(f32::ceil)
    }

    pub fn fract(&self) -> Val {
        self.map(f32::fract)
    }

    pub fn pow(&self, exp: Self) -> Result<Val, String> {
        match (self, exp) {
            (Float(_), Float(_))
                | (Vec2(_, _), Vec2(_, _))
                | (Vec3(_, _, _), Vec3(_, _, _))
                | (Vec4(_, _, _, _), Vec4(_, _, _, _))
            => Ok(self.zipmap(exp, |x, exp| x.powf(exp))),

            (Vec2(_, _), Float(exp))
                | (Vec3(_, _, _), Float(exp))
                | (Vec4(_, _, _, _), Float(exp))
            => Ok(self.map(|x| x.powf(exp))),

            _ => Err(format!("Expected pow(float, float), pow(vec2, vec2), pow(vec3, vec3), pow(vec4, vec4), pow(vec2, float), pow(vec3, float), pow(vec4, float), got pow({:?}, {:?}", self, exp))
        }
    }

    pub fn dot(&self, o: Self) -> Result<Val, String> {
        match (self, o) {
            (Vec2(x0, y0), Vec2(x1, y1)) => Ok(Float(x0 * x1 + y0 * y1)),
            (Vec3(x0, y0, z0), Vec3(x1, y1, z1)) => Ok(Float(x0 * x1 + y0 * y1 + z0 * z1)),
            (Vec4(x0, y0, z0, w0), Vec4(x1, y1, z1, w1)) => {
                Ok(Float(x0 * x1 + y0 * y1 + z0 * z1 + w0 + w1))
            }
            _ => Err(format!(
                "Expected dot(vec2, vec2), dot(vec3, vec3), dot(vec4, vec4), got dot({:?}, {:?})",
                self, o
            )),
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

    pub fn min(&self, val: Self) -> Result<Val, String> {
        match (self, val) {
            (Float(_), Float(_))
                | (Vec2(_, _), Vec2(_, _))
                | (Vec3(_, _, _), Vec3(_, _, _))
                | (Vec4(_, _, _, _), Vec4(_, _, _, _))
            => Ok(self.zipmap(val, |x, y| x.min(y))),
            _ => Err(format!("Expected min(float, float), min(vec2, vec2), min(vec3, vec3), min(vec4, vec4), got min({:?}, {:?})", self, val))
        }
    }

    pub fn max(&self, val: Self) -> Result<Val, String> {
        match (self, val) {
            (Float(_), Float(_))
                | (Vec2(_, _), Vec2(_, _))
                | (Vec3(_, _, _), Vec3(_, _, _))
                | (Vec4(_, _, _, _), Vec4(_, _, _, _))
            => Ok(self.zipmap(val, |x, y| x.max(y))),
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
            Vec4(x, y, z, w) => x + y + z + w,

            _ => panic!("Tried to get length of matrix"),
        };

        Float(sum.sqrt())
    }

    pub fn dist(&self, o: Self) -> Result<Val, String> {
        match (self, o) {
            (Float(_), Float(_))
                | (Vec2(_, _), Vec2(_, _))
                | (Vec3(_, _, _), Vec3(_, _, _))
                | (Vec4(_, _, _, _), Vec4(_, _, _, _))
            => Ok(self.zipmap(o, |x, y| x - y).length()),
            _ => Err(format!("Expected dist(float, float), dist(vec2, vec2), dist(vec3, vec3), dist(vec4, vec4), got dist({:?}, {:?})", self, o))
        }
    }

    pub fn cross(&self, o: Self) -> Result<Val, String> {
        match (self, o) {
            (Vec3(x0, y0, z0), Vec3(x1, y1, z1)) => Ok(Vec3(
                y0 * z1 - z0 * y1,
                z0 * x1 - x0 * z1,
                x0 * y1 - y0 * x1,
            )),
            _ => Err(format!(
                "Expected cross(vec3, vec3), got cross({:?}, {:?})",
                self, o
            )),
        }
    }

    pub fn norm(&self) -> Val {
        self.zipmap(self.length(), |x, len| x / len)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BuiltIn {
    // image utilities
    Sample,

    // vector utilities
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
    Length,
    Dot,
    Cross,
    Norm,

    // matrix utilities
    Mat2,
    Mat3,
    Mat4,
    RotateX,
    RotateY,
    RotateZ,
    Rotate,
    Scale,
    Translate,
    Ortho,
    LookAt,
    Perspective
}

#[derive(Debug, Clone, PartialEq)]
pub enum Ast {
    V(Val),
    Repeat(f32, Box<SrcAst>),
    While(Box<SrcAst>, f32, Box<SrcAst>),
    Assign(Spur, Box<SrcAst>),
    Block(Vec<SrcAst>),
    VecLiteral(
        Box<SrcAst>,
        Box<SrcAst>,
        Option<Box<SrcAst>>,
        Option<Box<SrcAst>>,
    ),
    VecRepeated(Box<SrcAst>, Box<SrcAst>),
    VecAccess(Box<SrcAst>, Swizzle),
    MatAccess(Box<SrcAst>, Box<SrcAst>),
    Ident(Spur),
    Return(Box<SrcAst>),
    Give(Box<SrcAst>),
    BinOp(Box<SrcAst>, Op, Box<SrcAst>),
    If {
        cond: Box<SrcAst>,
        true_ret: Box<SrcAst>,
        false_ret: Box<SrcAst>,
    },
    Call(BuiltIn, Vec<SrcAst>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct SrcAst {
    pub ast: Ast,
    pub line: usize,
}
impl SrcAst {
    //! had to resist an overwhelming urge to name this "mapro"
    pub fn map<F: Fn(SrcAst) -> SrcAst>(self, f: &F) -> Self {
        use Ast::*;
        let map = |sa: SrcAst| sa.map(f);
        let bmap = |sa: Box<SrcAst>| Box::new(sa.map(f));

        let SrcAst { ast, line } = f(self);
        SrcAst {
            line,
            ast: match ast {
                V(v) => V(v),
                Assign(s, t) => Assign(s, bmap(t)),
                While(cond, n, block) => While(bmap(cond), n, bmap(block)),
                Repeat(n, x) => Repeat(n, bmap(x)),
                Block(nodes) => Block(nodes.into_iter().map(map).collect()),
                VecLiteral(x, y, z, w) => {
                    VecLiteral(bmap(x), bmap(y), z.map(|z| bmap(z)), w.map(|w| bmap(w)))
                }
                VecRepeated(l, n) => VecRepeated(bmap(l), bmap(n)),
                VecAccess(v, s) => VecAccess(bmap(v), s),
                MatAccess(v, f) => MatAccess(bmap(v), bmap(f)),
                Ident(s) => Ident(s),
                Return(expr) => Return(bmap(expr)),
                Give(expr) => Give(bmap(expr)),
                BinOp(lhs, op, rhs) => BinOp(bmap(lhs), op, bmap(rhs)),
                If {
                    cond,
                    true_ret: t,
                    false_ret: f,
                } => If {
                    cond: bmap(cond),
                    true_ret: bmap(t),
                    false_ret: bmap(f),
                },
                Call(bi, args) => Call(bi, args.into_iter().map(map).collect()),
            },
        }
    }
}

impl SrcAst {
    pub fn new(ast: Ast, line: usize) -> SrcAst {
        SrcAst { ast, line }
    }
}

#[derive(Debug, Clone)]
pub struct Sampler<'a> {
    image: &'a ImageBuffer<Rgba<u8>, Vec<u8>>,
    width: u32,
    height: u32
}

impl<'a> Sampler<'a> {
    pub fn from(image: &'a ImageBuffer<Rgba<u8>, Vec<u8>>) -> Sampler {
        let (width, height) = image.dimensions();

        Sampler { image, width, height }
    }

    pub fn sample(&self, x: f32, y: f32) -> Val {
        if x > 1.0 || x < 0.0 {
            return Vec4(0.0, 0.0, 0.0, 0.0);
        }
        
        if y > 1.0 || y < 0.0 {
            return Vec4(0.0, 0.0, 0.0, 0.0);
        }

        let [r, g, b, a] = self.image.get_pixel((x * ((self.width - 1) as f32)) as u32, (y * ((self.height - 1) as f32)) as u32).0;

        Vec4(r as f32 / 255.0, g as f32 / 255.0,  b as f32 / 255.0, a as f32 / 255.0)
    }
}

#[derive(Default, Debug)]
pub struct Env<'a> {
    vars: HashMap<Spur, Val>,
    sampler: Option<Sampler<'a>>,
    pub ret: Option<Val>,
    parent: Option<Box<Env<'a>>>,
}

impl<'a> Env<'a> {
    pub fn with_sampler(sampler: Sampler) -> Env {
        let mut env = Env::default();

        env.sampler = Some(sampler);

        env
    }

    fn child(self) -> Self {
        Self {
            vars: Default::default(),
            sampler: self.sampler.clone(),
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

pub struct EvalRet<'a> {
    pub env: Env<'a>,
    pub val: Option<Val>,
    give: Option<Val>,
}

struct ERVal<'a> {
    env: Env<'a>,
    val: Val,
}

impl<'a> EvalRet<'a> {
    fn new(env: Env<'a>) -> Self {
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

    fn needs_val(self) -> ERVal<'a> {
        ERVal {
            env: self.env,
            val: need_val(self.val),
        }
    }
}

fn need_val(v: Option<Val>) -> Val {
    v.expect("context requires associated expression to return a value")
}

pub fn is_const(SrcAst { ast, .. }: &SrcAst) -> bool {
    use Ast::*;

    match ast {
        &V(_) => true,
        Assign(_, _) => false,
        While(cond, _, block) => is_const(cond) && is_const(block),
        Repeat(_, e) => is_const(e),
        Give(_) => false,
        Block(nodes) => matches!(
            nodes.iter().skip_while(|x| is_const(x)).next(),
            Some(SrcAst{ ast: Give(x), .. }) if is_const(x)
        ),
        VecLiteral(x, y, z, w) => {
            is_const(x)
                && is_const(y)
                && z.as_deref().map(is_const).unwrap_or(true)
                && w.as_deref().map(is_const).unwrap_or(true)
        }
        VecRepeated(l, n) => is_const(l) && is_const(n),
        VecAccess(v, _) => is_const(v),
        MatAccess(v, x) => is_const(v) && is_const(x),
        Ident(_) => false,
        Return(_) => false,
        BinOp(lhs, _, rhs) => is_const(lhs) && is_const(rhs),
        If {
            cond,
            true_ret: t,
            false_ret: f,
        } => is_const(cond) && is_const(t) && is_const(f),
        Call(_, args) => args.into_iter().all(is_const),
    }
}

pub fn fold(SrcAst { ast, .. }: &SrcAst) -> Option<Val> {
    use Ast::*;

    match ast {
        &V(v) => Some(v),
        Assign(_, _) => None,
        Block(nodes) => nodes
            .iter()
            .fold(Some(None::<Val>), |a, x| match (a, &x.ast) {
                (Some(_), Give(g)) => Some(fold(g)),
                _ => a.filter(|_| fold(x).is_some()),
            })
            .flatten(),
        VecLiteral(x, y, m_z, m_w) => Some(
            match (
                fold(x).and_then(|v| v.float()),
                fold(y).and_then(|v| v.float()),
                m_z.as_ref().map(|z| fold(z).and_then(|v| v.float())),
                m_w.as_ref().map(|w| fold(w).and_then(|v| v.float())),
            ) {
                (Some(x), Some(y), Some(Some(z)), Some(Some(w))) => Val::Vec4(x, y, z, w),
                (Some(x), Some(y), Some(Some(z)), None) => Val::Vec3(x, y, z),
                (Some(x), Some(y), None, None) => Val::Vec2(x, y),
                _ => return None,
            },
        ),
        _ => None, /*
                   VecRepeated(Box<SrcAst>, Box<SrcAst>),
                   VecAccess(Box<SrcAst>, Swizzle),
                   Ident(Spur),
                   Return(Box<SrcAst>),
                   Give(Box<SrcAst>),
                   BinOp(Box<SrcAst>, Op, Box<SrcAst>),
                   If {
                       cond: Box<SrcAst>,
                       true_ret: Box<SrcAst>,
                       false_ret: Box<SrcAst>,
                   },
                   Call(BuiltIn, Vec<SrcAst>), */
    }
}

macro_rules! builtin_one_arg {
    ($op:path, $function:expr, $len:ident, $line:ident, $vals:ident, $env:ident) => {{
        if $len != 1 {
            report(
                ErrorType::Runtime,
                $line,
                format!("Expected 1 input to {}(a), got {}", $function, $len),
            )
        }

        EvalRet::new($env).with_val(Some($op(&$vals.pop().unwrap())))
    }};
}

macro_rules! builtin_two_args {
    ($op:path, $function:expr, $len:ident, $line:ident, $vals:ident, $env:ident) => {{
        if $len != 2 {
            report(
                ErrorType::Runtime,
                $line,
                format!("Expected 2 inputs to {}(a, b), got {}", $function, $len),
            )
        }

        let arg2 = $vals.pop().unwrap();
        let arg1 = $vals.pop().unwrap();

        EvalRet::new($env).with_val(Some(match $op(&arg1, arg2) {
            Ok(result) => result,
            Err(error) => report(ErrorType::Runtime, $line, error.as_str()),
        }))
    }};
}

macro_rules! builtin_three_args {
    ($op:path, $function:expr, $len:ident, $line:ident, $vals:ident, $env:ident) => {{
        if $len != 3 {
            report(
                ErrorType::Runtime,
                $line,
                format!("Expected 3 inputs to {}(a, b, c), got {}", $function, $len),
            )
        }

        let arg3 = $vals.pop().unwrap();
        let arg2 = $vals.pop().unwrap();
        let arg1 = $vals.pop().unwrap();

        EvalRet::new($env).with_val(Some(match $op(&arg1, arg2, arg3) {
            Ok(result) => result,
            Err(error) => report(ErrorType::Runtime, $line, error.as_str()),
        }))
    }};
}

pub fn eval<'a>(&SrcAst { line, ref ast }: &SrcAst, e: Env<'a>, r: &Rodeo) -> EvalRet<'a> {
    use Ast::*;

    match ast {
        Repeat(times, block) => {
            if *times < 1.0 - f32::EPSILON {
                report(ErrorType::Runtime, line, format!("Expected positive compile-time literal in repeat statement, got {}", times));
            }

            let t = *times as usize;

            if (times - (t as f32)).abs() > f32::EPSILON {
                report(ErrorType::Runtime, line, format!("Expected whole number in repeat statement, got {}", times));
            }

            let mut ret = eval(block, e, &r);

            for _ in 0..t - 1 {
                ret = eval(block, ret.env, &r);
            }

            ret
        }
        While(cond, times, block) => {
            if *times < 1.0 - f32::EPSILON {
                report(ErrorType::Runtime, line, format!("Expected positive number of iterations in while statement, got {}", times));
            }

            let t = *times as usize;

            if (times - (t as f32)).abs() > f32::EPSILON {
                report(ErrorType::Runtime, line, format!("Expected whole number of iterations in while statement, got {}", times));
            }

            let mut env = e;

            for _ in 0..t {
                let ERVal { env: e, val } = eval(cond, env, r).needs_val();
                env = e;

                match val {
                    Float(condition) => if condition != 1.0 {
                        break
                    },
                    x => report(ErrorType::Runtime, line, format!("Expected condition to evaluate to float in while statement, got {:?}", x))
                }

                env = eval(block, env, r).env;
            }

            EvalRet::new(env)
       }
        &V(v) => EvalRet::new(e).with_val(Some(v)),
        Assign(i, to) => {
            let ERVal { mut env, val } = eval(&to, e, r).needs_val();
            env.set(*i, val);
            EvalRet::new(env).with_val(Some(val))
        }
        Block(nodes) => {
            let EvalRet { env, give, .. } =
                nodes.iter().fold(EvalRet::new(e.child()), |acc, node| {
                    if acc.give.or(acc.env.ret).is_some() {
                        acc
                    } else {
                        eval(node, acc.env, r)
                    }
                });
            EvalRet::new(*env.parent.unwrap()).with_val(env.ret.or(give))
        }
        VecRepeated(value, length) => {
            let ERVal { val: value, env } = eval(value, e, r).needs_val();
            let ERVal { val: length, env } = eval(length, env, r).needs_val();

            match (value, length) {
                (Float(v), Float(l)) => match l as usize {
                    2 => EvalRet::new(env).with_val(Some(Val::Vec2(v, v))),
                    3 => EvalRet::new(env).with_val(Some(Val::Vec3(v, v, v))),
                    4 => EvalRet::new(env).with_val(Some(Val::Vec4(v, v, v, v))),
                    n => {
                        panic!("{} is an invalid vector length; must be 2 <= x <= 4", n)
                    }
                },
                _ => report(
                    ErrorType::Runtime,
                    line,
                    "Both X and L in [X; L] in vector literal must evaluate to scalars",
                ),
            }
        }
        VecLiteral(xast, yast, None, None) => {
            let ERVal { val: xval, env } = eval(&xast, e, r).needs_val();
            let ERVal { val: yval, env } = eval(&yast, env, r).needs_val();
            match (xval, yval) {
                (Float(x), Float(y)) => EvalRet::new(env).with_val(Some(Val::Vec2(x, y))),
                _ => panic!(),
            }
        }
        VecLiteral(_, _, None, Some(_)) => panic!("fuck you"),
        VecLiteral(xast, yast, Some(zast), None) => {
            let ERVal { val: xval, env } = eval(&xast, e, r).needs_val();
            let ERVal { val: yval, env } = eval(&yast, env, r).needs_val();
            let ERVal { val: zval, env } = eval(&zast, env, r).needs_val();
            match (xval, yval, zval) {
                (Float(x), Float(y), Float(z)) => {
                    EvalRet::new(env).with_val(Some(Val::Vec3(x, y, z)))
                }
                _ => panic!(),
            }
        }
        VecLiteral(xast, yast, Some(zast), Some(wast)) => {
            let ERVal { val: xval, env } = eval(&xast, e, r).needs_val();
            let ERVal { val: yval, env } = eval(&yast, env, r).needs_val();
            let ERVal { val: zval, env } = eval(&zast, env, r).needs_val();
            let ERVal { val: wval, env } = eval(&wast, env, r).needs_val();
            match (xval, yval, zval, wval) {
                (Float(x), Float(y), Float(z), Float(w)) => {
                    EvalRet::new(env).with_val(Some(Val::Vec4(x, y, z, w)))
                }
                _ => panic!(),
            }
        }
        MatAccess(mat, row) => {
            let ERVal { env, val: matrix } = eval(mat, e, r).needs_val();
            let ERVal { env, val: access } = eval(row, env, r).needs_val();

            let access = if let Float(access) = access {
                access
            } else {
                report(
                    ErrorType::Runtime,
                    line,
                    format!("Expected float when indexing matrix, got {:?}", access),
                )
            };

            let row = match matrix {
                Mat2(_, _) => matrix.index_matrix(access as usize),
                Mat3(_, _, _) => matrix.index_matrix(access as usize),
                Mat4(_, _, _, _) => matrix.index_matrix(access as usize),
                _ => report(
                    ErrorType::Runtime,
                    line,
                    format!("Expected matrix when indexing, got {:?}", matrix),
                ),
            };

            match row {
                Ok(row) => return EvalRet::new(env).with_val(Some(row)),
                Err(error) => report(ErrorType::Runtime, line, error.as_str()),
            }
        }
        VecAccess(access_me, swiz) => {
            let ERVal { env, val } = eval(&access_me, e, r).needs_val();

            if let Val::Float(x) = val {
                report(
                    ErrorType::Runtime,
                    line,
                    format!("Expected vector when swizzling, got {}", x),
                );
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
                _ => report(ErrorType::Runtime, line, "Invalid swizzle"),
            }))
        }
        &Ident(i) => {
            if let Some(val) = e.get(i) {
                return EvalRet::new(e).with_val(Some(val));
            }

            report(
                ErrorType::Runtime,
                line,
                format!("Couldn't resolve identifier '{}'", r.resolve(&i)),
            )
        }
        Return(v) => {
            let ERVal { mut env, val } = eval(&v, e, r).needs_val();
            env.return_val(val);
            EvalRet::new(env)
        }
        Give(g) => {
            let ERVal { env, val } = eval(&g, e, r).needs_val();
            EvalRet::new(env).with_give(val)
        }
        BinOp(lhs, op, rhs) => {
            let ERVal { env, val: lval } = eval(&lhs, e, r).needs_val();
            let ERVal { env, val: rval } = eval(&rhs, env, r).needs_val();

            use Val::*;
            EvalRet::new(env).with_val(Some(match (lval, op, rval) {
                (Float(_), Op::Sub, Vec2(_, _) | Vec3(_, _, _) | Vec4(_, _, _, _)) => {
                    report(ErrorType::Runtime, line, "Expected vector - float, got float - vector");
                }
                (Float(_), Op::Add, Vec2(_, _) | Vec3(_, _, _) | Vec4(_, _, _, _)) => {
                    report(ErrorType::Runtime, line, "Expected vector + float, got float + vector");
                }
                (Float(_), Op::Div, Vec2(_, _) | Vec3(_, _, _) | Vec4(_, _, _, _)) => {
                    report(ErrorType::Runtime, line, "Expected vector / float, got float / vector")
                }
                (_, Op::Sub, _) => lval.zipmap(rval, |l, r| l - r),
                (_, Op::Add, _) => lval.zipmap(rval, |l, r| l + r),
                (_, Op::Mul, _) => match lval.mult(rval) {
                    Ok(result) => result,
                    Err(error) => report(ErrorType::Runtime, line, error.as_str())
                }
                (_, Op::Div, _) => lval.zipmap(rval, |l, r| l / r),
                (Float(l), Op::More, Float(r)) => Float((l > r) as i32 as f32),
                (Float(l), Op::Less, Float(r)) => Float((l < r) as i32 as f32),
                (Float(l), Op::MoreEq, Float(r)) => Float((l >= r) as i32 as f32),
                (Float(l), Op::LessEq, Float(r)) => Float((l <= r) as i32 as f32),
                (Float(l), Op::Equal, Float(r)) => Float(if l == r { 1.0 } else { 0.0 }),
                _ => report(
                    ErrorType::Runtime,
                    line,
                    format!("Unexpected scalar/vector binary operation relationship, got {:#?} {:#?} {:#?}", lval, op, rval),
                )
            }))
        }
        If {
            cond,
            true_ret,
            false_ret,
        } => {
            let ERVal { env, val: condval } = eval(&cond, e, r).needs_val();
            match condval {
                Val::Float(f) if f == 1.0 => eval(&true_ret, env, r),
                Val::Float(_) => eval(&false_ret, env, r),
                _ => report(
                    ErrorType::Runtime,
                    line,
                    format!(
                        "Expected scalar conditional in if expression, got {:#?}",
                        condval
                    )
                    .as_str(),
                ),
            }
        }
        Call(builtin, args) => {
            let len = args.len();
            let (env, mut vals) =
                args.iter()
                    .fold((e, Vec::with_capacity(len)), |(env, mut vals), arg| {
                        let ERVal { env, val } = eval(arg, env, r).needs_val();
                        vals.push(val);
                        (env, vals)
                    });

            match *builtin {
                BuiltIn::Mat2 => {
                    if len != 2 {
                        report(
                            ErrorType::Runtime,
                            line,
                            format!("Expected 2 inputs to mat2(vec2, vec2), got {}", len),
                        );
                    }

                    let arg2 = vals.pop().unwrap();
                    let arg1 = vals.pop().unwrap();

                    let mat = match (arg1, arg2) {
                        (Vec2(x0, y0), Vec2(x1, y1)) => Mat2([x0, y0], [x1, y1]),
                        (x, y) => report(
                            ErrorType::Runtime,
                            line,
                            format!("Expected mat2(vec2, vec2), got mat2({:?}, {:?})", x, y)
                                .as_str(),
                        ),
                    };

                    EvalRet::new(env).with_val(Some(mat))
                }
                BuiltIn::Mat3 => {
                    if len != 3 {
                        report(
                            ErrorType::Runtime,
                            line,
                            format!("Expected 3 inputs to mat3(vec3, vec3, vec3), got {}", len)
                                .as_str(),
                        );
                    }

                    let arg3 = vals.pop().unwrap();
                    let arg2 = vals.pop().unwrap();
                    let arg1 = vals.pop().unwrap();

                    let mat = match (arg1, arg2, arg3) {
                        (Vec3(x0, y0, z0), Vec3(x1, y1, z1), Vec3(x2, y2, z2)) => {
                            Mat3([x0, y0, z0], [x1, y1, z1], [x2, y2, z2])
                        }
                        (x, y, z) => report(
                            ErrorType::Runtime,
                            line,
                            format!(
                                "Expected mat3(vec3, vec3, vec3), got mat3({:?}, {:?}, {:?})",
                                x, y, z
                            )
                            .as_str(),
                        ),
                    };

                    EvalRet::new(env).with_val(Some(mat))
                }
                BuiltIn::Mat4 => {
                    if len != 4 {
                        report(
                            ErrorType::Runtime,
                            line,
                            format!(
                                "Expected 4 inputs to mat4(vec4, vec4, vec4, vec4), got {}",
                                len
                            )
                            .as_str(),
                        );
                    }

                    let arg4 = vals.pop().unwrap();
                    let arg3 = vals.pop().unwrap();
                    let arg2 = vals.pop().unwrap();
                    let arg1 = vals.pop().unwrap();

                    let mat = match (arg1, arg2, arg3, arg4) {
                        (Vec4(x0, y0, z0, w0), Vec4(x1, y1, z1, w1), Vec4(x2, y2, z2, w2), Vec4(x3, y3, z3, w3)) => Mat4([x0, y0, z0, w0], [x1, y1, z1, w1], [x2, y2, z2, w2], [x3, y3, z3, w3]),
                        (x, y, z, w) => report(ErrorType::Runtime, line, format!("Expected mat4(vec4, vec4, vec4, vec4), got mat4({:?}, {:?}, {:?}, {:?})", x, y, z, w))
                    };

                    EvalRet::new(env).with_val(Some(mat))
                }
                BuiltIn::Sample => {
                    if len != 1 {
                        report(
                            ErrorType::Runtime,
                            line,
                            format!(
                                "Expected 1 input to sample(vec2), got {}",
                                len
                            )
                            .as_str(),
                        );
                    }

                    let pixel = match (vals.pop().unwrap(), &env.sampler) {
                        (Vec2(x, y), Some(sampler)) => sampler.sample(x, y),
                        (x, _) => report(ErrorType::Runtime, line, format!("Expected sample(vec2), got sample({:?}", x))
                    };

                    EvalRet::new(env).with_val(Some(pixel))
                }
                BuiltIn::Dist => {
                    builtin_two_args!(Val::dist, "dist", len, line, vals, env)
                }
                BuiltIn::Pow => {
                    builtin_two_args!(Val::pow, "pow", len, line, vals, env)
                }
                BuiltIn::Sin => {
                    builtin_one_arg!(Val::sin, "sin", len, line, vals, env)
                }
                BuiltIn::Cos => {
                    builtin_one_arg!(Val::cos, "cos", len, line, vals, env)
                }
                BuiltIn::Tan => {
                    builtin_one_arg!(Val::tan, "tan", len, line, vals, env)
                }
                BuiltIn::Asin => {
                    builtin_one_arg!(Val::asin, "asin", len, line, vals, env)
                }
                BuiltIn::Acos => {
                    builtin_one_arg!(Val::acos, "acos", len, line, vals, env)
                }
                BuiltIn::Atan => {
                    builtin_one_arg!(Val::atan, "atan", len, line, vals, env)
                }
                BuiltIn::Exp => {
                    builtin_one_arg!(Val::exp, "exp", len, line, vals, env)
                }
                BuiltIn::Log => {
                    builtin_one_arg!(Val::log, "log", len, line, vals, env)
                }
                BuiltIn::Sqrt => {
                    builtin_one_arg!(Val::sqrt, "sqrt", len, line, vals, env)
                }
                BuiltIn::InverseSqrt => {
                    builtin_one_arg!(Val::invsqrt, "invsqrt", len, line, vals, env)
                }
                BuiltIn::Abs => {
                    builtin_one_arg!(Val::abs, "abs", len, line, vals, env)
                }
                BuiltIn::Sign => {
                    builtin_one_arg!(Val::sign, "sign", len, line, vals, env)
                }
                BuiltIn::Floor => {
                    builtin_one_arg!(Val::floor, "floor", len, line, vals, env)
                }
                BuiltIn::Ceil => {
                    builtin_one_arg!(Val::ceil, "ceil", len, line, vals, env)
                }
                BuiltIn::Fract => {
                    builtin_one_arg!(Val::fract, "fract", len, line, vals, env)
                }
                BuiltIn::Mod => {
                    builtin_two_args!(Val::modulo, "mod", len, line, vals, env)
                }
                BuiltIn::Min => {
                    builtin_two_args!(Val::min, "min", len, line, vals, env)
                }
                BuiltIn::Max => {
                    builtin_two_args!(Val::max, "max", len, line, vals, env)
                }
                BuiltIn::Clamp => {
                    builtin_three_args!(Val::clamp, "clamp", len, line, vals, env)
                }
                BuiltIn::Mix => {
                    builtin_three_args!(Val::mix, "mix", len, line, vals, env)
                }
                BuiltIn::Step => {
                    builtin_two_args!(Val::step, "step", len, line, vals, env)
                }
                BuiltIn::Length => {
                    builtin_one_arg!(Val::length, "length", len, line, vals, env)
                }
                BuiltIn::Dot => {
                    builtin_two_args!(Val::dot, "dot", len, line, vals, env)
                }
                BuiltIn::Cross => {
                    builtin_two_args!(Val::cross, "cross", len, line, vals, env)
                }
                BuiltIn::Norm => {
                    builtin_one_arg!(Val::norm, "norm", len, line, vals, env)
                }
                BuiltIn::Radians => {
                    builtin_one_arg!(Val::radians, "radians", len, line, vals, env)
                }
                BuiltIn::Degrees => {
                    builtin_one_arg!(Val::degrees, "degrees", len, line, vals, env)
                }
                BuiltIn::RotateX => {
                    if len != 1 {
                        report(ErrorType::Runtime, line, format!("Expected 1 input to rotate_x(a), got {}", len));
                    }

                    match vals.pop().unwrap() {
                        Float(radians) => EvalRet::new(env).with_val(Some(Val::rotate_x(radians))),
                        x => report(ErrorType::Runtime, line, format!("Expected rotate_x(float), got rotate_x({:?})", x))
                    }
                }
                BuiltIn::RotateY => {
                    if len != 1 {
                        report(ErrorType::Runtime, line, format!("Expected 1 input to rotate_y(a), got {}", len));
                    }

                    match vals.pop().unwrap() {
                        Float(radians) => EvalRet::new(env).with_val(Some(Val::rotate_y(radians))),
                        x => report(ErrorType::Runtime, line, format!("Expected rotate_y(float), got rotate_y({:?})", x))
                    }
                },
                BuiltIn::RotateZ => {
                    if len != 1 {
                        report(ErrorType::Runtime, line, format!("Expected 1 input to rotate_z(a), got {}", len));
                    }

                    match vals.pop().unwrap() {
                        Float(radians) => EvalRet::new(env).with_val(Some(Val::rotate_z(radians))),
                        x => report(ErrorType::Runtime, line, format!("Expected rotate_z(float), got rotate_z({:?})", x))
                    }
                },
                BuiltIn::Rotate => {
                    if len != 3 {
                        report(ErrorType::Runtime, line, format!("Expected 3 inputs to rotate(a, b, c), got {}", len));
                    }

                    let arg3 = vals.pop().unwrap();
                    let arg2 = vals.pop().unwrap();
                    let arg1 = vals.pop().unwrap();

                    match (arg1, arg2, arg3) {
                        (Float(yaw), Float(pitch), Float(roll)) => EvalRet::new(env).with_val(Some(Val::rotate(yaw, pitch, roll))),
                        (x, y, z) => report(ErrorType::Runtime, line, format!("Expected rotate(float, float, float), got rotate({:?}, {:?}, {:?})", x, y, z))
                    }
                },
                BuiltIn::Scale => {
                    if len != 1 {
                        report(ErrorType::Runtime, line, format!("Expected 1 input to scale(a), got {}", len));
                    }

                    match vals.pop().unwrap() {
                        Float(s) => EvalRet::new(env).with_val(Some(Val::scale(s))),
                        x => report(ErrorType::Runtime, line, format!("Expected scale(float), got scale({:?})", x))
                    }
                }
                BuiltIn::Translate => {
                    if len != 3 {
                        report(ErrorType::Runtime, line, format!("Expected 3 inputs to translate(a, b, c), got {}", len));
                    }

                    let arg3 = vals.pop().unwrap();
                    let arg2 = vals.pop().unwrap();
                    let arg1 = vals.pop().unwrap();

                    match (arg1, arg2, arg3) {
                        (Float(x), Float(y), Float(z)) => EvalRet::new(env).with_val(Some(Val::translate(x, y, z))),
                        (x, y, z) => report(ErrorType::Runtime, line, format!("Expected translate(float, float, float), got translate({:?}, {:?}, {:?})", x, y, z))
                    }
                },
                BuiltIn::Ortho => {
                    if len != 6 {
                        report(ErrorType::Runtime, line, format!("Expected 6 inputs to ortho(a, b, c, d, e, f), got {}", len));
                    }

                    let arg6 = vals.pop().unwrap();
                    let arg5 = vals.pop().unwrap();
                    let arg4 = vals.pop().unwrap();
                    let arg3 = vals.pop().unwrap();
                    let arg2 = vals.pop().unwrap();
                    let arg1 = vals.pop().unwrap();

                    match (arg1, arg2, arg3, arg4, arg5, arg6) {
                        (Float(near), Float(far), Float(left), Float(right), Float(top), Float(bottom)) => EvalRet::new(env).with_val(Some(Val::ortho(near, far, left, right, top, bottom))),
                        (x, y, z, w, v, u) => report(ErrorType::Runtime, line, format!("Expected ortho(float, float, float, float, float, float), got ortho({:?}, {:?}, {:?}, {:?}, {:?}, {:?})", x, y, z, w, v, u))
                    }
                }
                BuiltIn::LookAt => {
                    if len != 2 {
                        report(ErrorType::Runtime, line, format!("Expected 2 inputs to lookat(a, b), got {}", len));
                    }

                    let arg2 = vals.pop().unwrap();
                    let arg1 = vals.pop().unwrap();
    
                    match (arg1, arg2) {
                        (Vec3(_, _, _), Vec3(_, _, _)) => EvalRet::new(env).with_val(Some(Val::lookat(arg1, arg2))),
                        (x, y) => report(ErrorType::Runtime, line, format!("Expected lookat(vec3, vec3), got lookat({:?}, {:?})", x, y))
                    }
                }
                BuiltIn::Perspective => {
                    if len != 2 {
                        report(ErrorType::Runtime, line, format!("Expected 2 inputs to lookat(a, b), got {}", len));
                    }

                    let arg2 = vals.pop().unwrap();
                    let arg1 = vals.pop().unwrap();
    
                    match (arg1, arg2) {
                        (Vec3(_, _, _), Vec3(_, _, _)) => EvalRet::new(env).with_val(Some(Val::lookat(arg1, arg2))),
                        (x, y) => report(ErrorType::Runtime, line, format!("Expected lookat(vec3, vec3), got lookat({:?}, {:?})", x, y))
                    }
                }
            }
        }
    }
}

#[test]
fn constant_folding() {
    assert_eq!(
        crate::string_to_ast("return [1 + 2; 4]").1.ast,
        crate::string_to_ast("return [3; 4]").1.ast,
    );

    assert_eq!(
        crate::string_to_ast("return { give [1 + 2; 4] }").1.ast,
        crate::string_to_ast("return [3; 4]").1.ast,
    );
}
