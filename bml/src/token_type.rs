#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum TokenType {
    // single-char tokens
    LeftParen,
    RightParen,
    Comma,
    LeftSquare,
    RightSquare,
    LeftBracket,
    RightBracket,
    Semi,
    Equals,

    // one-or-two char tokens
    Plus,
    Minus,
    Slash,
    Star,
    NotEquals,
    EqualsEquals,
    GreaterThan,
    LessThan,
    LessThanEquals,
    GreaterThanEquals,

    // swizzling
    Dot,
    X,
    Y,
    Z,
    W,

    // keywords
    If,
    Else,
    Return,
    Give,
    Macro,
    Repeat,
    While,

    // literals
    Number,
    Vec2,
    Vec3,
    Vec4,

    // identifiers
    Identifier,

    Eof,
}
