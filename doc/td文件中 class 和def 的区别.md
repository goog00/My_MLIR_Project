在 MLIR 的 TableGen (`.td`) 文件中，`class` 和 `def` 是 **两种核心构造**，但它们的作用不同：

---

### ✅ `class` —— **模板定义（模板类）**

* 类似于 C++ 中的模板/类。
* 不会直接生成实体，而是用于 **复用、继承、参数化定义**。
* 可以包含 `let` 字段、继承、默认值、参数等。
* 必须通过 `def` 或其他 `class` 派生使用。

#### 示例：

```tablegen
class DreamStar_UnaryOp<string mnemonic, Type OperandType> 
    : DreamStar_Op<mnemonic> {
  let arguments = (ins OperandType:$input);
  let results = (outs OperandType:$result);
}
```

这是定义一个**一元操作模板类**，不能直接用于 IR 生成，但可以被 `def` 使用：

---

### ✅ `def` —— **实体定义（实例）**

* 类似于 C++ 中的对象或具体类定义。
* 会被 TableGen 编译为实际的操作、类型、属性、转换等生成代码。
* 用于定义具体的算子、属性、类型等。

#### 示例：

```tablegen
def DreamStar_ExpOp : DreamStar_UnaryOp<"exp", AnyDSTensor>;
```

这会实例化上面的 `DreamStar_UnaryOp` 类，生成一个名为 `exp` 的 MLIR 操作。

---

### 🧠 类比理解：

| TableGen 元素 | 类比于 C++        | 是否生成实际代码 |
| ----------- | -------------- | -------- |
| `class`     | 类、模板（Template） | ❌ 不直接生成  |
| `def`       | 类实例、对象（实例化）    | ✅ 会生成代码  |

---

### 📌 总结：

* **`class`：用于复用（模板），本身不产生代码。**
* **`def`：具体定义（实例），会生成 IR 操作/类型等。**

---

如果你正在编写多个相似的操作（如 `add`, `mul`, `exp`），用 `class` 抽象出公共结构，再用多个 `def` 实例化它，是最推荐的做法。

需要我演示如何从 `class` 创建多个 `def` 实例，生成一批算子吗？
