from regressao_logistica import *
import unittest

class TestFuncaoAtivacao(unittest.TestCase):
    def test_sigmoid(self):
        #self.assertEquals(sigmoid(-0.8))
        self.assertListEqual(list(sigmoid.funcao(np.array([-0.8,-0.4,0.2]))),[0.31002551887238755,0.401312339887548,0.549833997312478])
        self.assertAlmostEqual(sigmoid.funcao(-0.4),0.401312339887548)
        self.assertAlmostEqual(sigmoid.funcao(0),0.5)
        self.assertAlmostEqual(sigmoid.funcao(0.2),0.549833997312478)

    def test_sigmoid_dz(self):
        a = sigmoid.dz_funcao(np.array([-0.8,-0.4,0.2]),np.array([-0.8,-0.4,0.2]),np.array([0,10,1]))
        arr_esp = [-0.8,-10.4,-0.8]
        for i,esp in enumerate(arr_esp):
            self.assertAlmostEqual(arr_esp[i],a[i])
        self.assertAlmostEqual(sigmoid.dz_funcao(-0.4,0.4,2),-2.4)
        self.assertAlmostEqual(sigmoid.dz_funcao(0,0,0)   ,0)
        self.assertAlmostEqual(sigmoid.dz_funcao(0.2,0,0) ,0.2)

class TestRegressaoLogistica(unittest.TestCase):



    def setUp(self):
        self.metodo = RegressaoLogistica(lambda z:1/(1+np.power(math.e,-z)),lambda a,z,y:a-y,100)

        self.metodo.arr_w = np.array([0.5,0.2,0.9,0.3])
        self.metodo.b = 0.2
        self.mat_x =         np.array([[0.2,0.1,0.3,0.5],#instancia 1
                                        [0.1,0.5,0.1,0.1],#instancia 2
                                        [0.9,0.1,0.4,0.7]#instancia 3
                                        ])
        self.arr_result_esperado_z = [0.5*0.2+0.2*0.1+0.9*0.3+0.3*0.5+0.2,
                                      0.5*0.1+0.2*0.5+0.9*0.1+0.3*0.1+0.2,
                                      0.5*0.9+0.2*0.1+0.9*0.4+0.3*0.7+0.2]


    def test_z(self):
        print(self.metodo.arr_w.shape)
        print(self.mat_x.shape)
        arr_z = self.metodo.z(self.mat_x)
        self.z_test(arr_z)

    def test_forward_propagation(self):

        self.metodo.forward_propagation(self.mat_x)
        self.z_test(self.metodo.arr_z)
        arr_sigmoid_esperado = [1/(1+math.e**(-z)) for z in self.arr_result_esperado_z]
        self.assertEqual(len(self.metodo.arr_a),3,"O vetor arr_a (ativações) deveria possuir 3 elementos (um resultado por instancia)")

        for i,a_i in enumerate(arr_sigmoid_esperado):
            self.assertAlmostEqual(a_i,self.metodo.arr_a[i],msg="a("+str(object=i)+") inesperado")

    def test_backward_propagation(self):
        arr_y = np.array([1,0,1])
        self.metodo.forward_propagation(self.mat_x)
        arr_a = self.metodo.arr_a
        gradiente = self.metodo.backward_propagation(arr_y)
        self.assertEqual(len(gradiente.arr_dw),len(self.metodo.arr_w),"O tamanho de arr_dw (gradiente) deve ser o mesmo que o tamanho do vetor de pesos")
        self.assertTrue(isinstance(gradiente.db, (int, float, complex)) and not isinstance(gradiente.db, bool),"gradiente.db deve possuir um valor numérico")

        arr_dw = np.zeros(len(gradiente.arr_dw))
        db = 0
        for i,y in enumerate(arr_y):
            a = arr_a[i]
            dz = a-y
            print("DZ("+str(i)+") - teste:"+str(dz))
            for j,x_j in enumerate(self.mat_x[i]):
                arr_dw[j] += x_j*dz
            db += dz
        arr_dw = arr_dw/3
        db = db/3
        for i,dw_esp in enumerate(arr_dw):
            self.assertAlmostEqual(dw_esp,gradiente.arr_dw[i])

        self.assertAlmostEqual(db,gradiente.db)

    def test_atualiza_pesos(self):
        self.metodo.arr_w = np.array([2,3,2,5])
        self.metodo.b=5
        self.metodo.gradiente = Gradiente(np.zeros(4),np.array([0.2,0.1,0.4,0.1]),0.3)
        self.metodo.atualiza_pesos(0.1)

        self.assertListEqual(list(self.metodo.arr_w),[1.98,2.99,1.96,4.99])
        self.assertEqual(self.metodo.b,4.97)

    def test_loss_function(self):
        self.metodo.arr_a = np.array([0.9,0.9,0.1,0.9,0.1,0.5,0.3])
        arr_y = np.array([1,1,1,0,0,0,1])
        self.assertAlmostEqual(0.9740531025496358, self.metodo.loss_function(arr_y))

    def test_fit(self):
        self.metodo.fit(self.mat_x,np.array([0,0,1]),0.1)
        arr_expected  = [1.2286229930132182,-0.41843816570061465,0.8410999350627626,0.3238109930963504]
        for i,expected in enumerate(arr_expected):
            self.assertAlmostEqual(expected, self.metodo.arr_w[i])
        


    def test_predict(self):
        self.mat_x[1] = -0.3
        self.assertListEqual(list(self.metodo.predict(self.mat_x)),[1,0,1])
    def z_test(self,arr_z):
        self.assertEqual(len(arr_z),3,"O vetor retonado por z deveria possuir 3 elementos (um resultado por instancia)")
        for i,z in enumerate(arr_z):
            self.assertEqual(z,self.arr_result_esperado_z[i],'Resultado inesperado para a instancia #{i}'.format(i=i)+\
                                                        "\nVetor de pesos:{pesos} b: {b}\n".format(pesos=self.metodo.arr_w,b=self.metodo.b)+\
                                                        "Vetor de atributos: {att}\n".format(att=self.mat_x[i])+\
                                                        "Resultado:{z} esperado:{esp}".format(z=z,esp=self.arr_result_esperado_z[i]))
if __name__ == "__main__":
    unittest.main()
