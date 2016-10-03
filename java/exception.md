
# 异常分类

所有异常，都继承自java.lang.Throwable类，包括Exception和Error。

其中，RuntimeException继承自Exception称为运行时异常，RuntimeException和Error及其子类称之为未受检的异常，其余继承自Exception的子类称之为受检查的异常。

受检的含义是，编译器强制要求开发人员在代码中进行显式的抛出和捕获，否则就会产生编译错误。

### RuntimeException

运行时异常，通常是程序错误，不应该被捕获处理。但应该适当被api抛出。

### CheckedException

Java编译器要求程序必须捕获或声明抛出这种异常。

### Error

JVM内部的严重问题，无法恢复，不需要编程人员处理。一般应用代码不会实现Error子类。

# 异常设计

在异常设计中，对于可恢复的情况使用受检查的异常，对编程错误使用运行时异常。

应该尽可能的避免使用受检查的异常。如果一个api的调用可能会发可恢复的错误，导致调用失败，一般有三种设计方式。

### 使用异常

声明一个受检查的异常，发生时抛出该异常，强迫客户端处理或抛出该异常。

优点：api的使用规范体现在api中。

缺点：增加客户端编码成本。

### 状态检测方法

通过另一个额外的方法，来检测调用方法是否可能成功。

例如：
```java
Iterator it = list.iterator();
while(it.hasNext()){
    it.next();
}
```
hasNext是状态检测方法，它保证了next的调用成功。

优点：相对于使用异常，它依靠逻辑语句控制流程，程序的可读性强。

缺点：在多线程并发环境下缺少外部同步，状态检测方法和调用之间不是原子性的，可能会发生异常。客户端错调用，比如只调用next导致错误。

### 可识别的返回值

调用返回数据对象的api，再发生错误时返回null，代表发生错误，客户端需要进行null的检查。

缺点：api的规范体现在其实现内部，而不是接口上，需要额外的文档说明。在某些情况下null不能代表可识别的错误，需要额外的状态设计，例如状态码。

# 使用标准异常

尽可能使用标准的异常，避免自己实现异常。

列举常用的异常：

|异常|使用场合|
|---|---|
|IllegalArgumentException|非null的参数值不正确|
|IllegalStateException|对于方法调用而言，对象状态不合适|
|NullPointerException|在禁止使用null的情况下可参数值为null|
|IndexOutOfBoundsException|下标参数值越界|
|ConcurrentModificationException|在禁止并发修改的情况下，检测到对象的并发修改|
|UnsupportedOperationException|对象不支持用户请求的方法|

# 异常转义

通常在进行高层抽象编码时，调用的底层api会抛出受检查的低层异常，这时候如果直接向高层的客户端抛出低层异常，那么这个异常就会暴露低层的实现。

例如，一个访问存储的api，底层可以是访问mysql或者mongo，那么如果直接抛出mysql定义的异常，那么实现细节就暴露给高层的客户端了。

异常转义，就是通过catch低层的异常，然后抛出高层定义的异常。

```java
try{
  //do something
}catch(MysqlTimeoutException e){
  throw new AppTimeoutException();
}
```
