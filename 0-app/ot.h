class OT {
protected:
    Role role;
    int id;
public:
    virtual void send(GPUBlock &m0, GPUBlock &m1) = 0;
    virtual GPUBlock recv(uint8_t b) = 0;
};
