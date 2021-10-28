from abc import ABCMeta, abstractclassmethod
import cv2
import numpy as np
import matplotlib.pyplot as plt
import extrator_POSICAO as posicao
import extrator_ALTURA as altura
import extrator_PROXIMIDADE as proximidade


class Libras(metaclass=ABCMeta):

    @abstractclassmethod
    def processamento(self):
        pass

class RedeNeural(Libras):
    def __init__(self, imagem,proto,peso) -> None:
        self.image = imagem
        self.proto = proto
        self.pesos = peso

    
    def modelo(self):
        print('carregando  modulos...')
        
        modelo = cv2.dnn.readNetFromCaffe(self.proto, self.pesos)

        return modelo

    def processamento(self):
        print("Processamento...")
        imagem = cv2.imread(self.image)
        imagem_copia = np.copy(imagem)

        imagem_largura = imagem.shape[1]
        imagem_altura = imagem.shape[0]
        proporcao = imagem_largura / imagem_altura
        entrada_blob = cv2.dnn.blobFromImage(imagem, 1.0 / 255, 
                                     (int(((proporcao * 256) * 8) // 8), 256), 
                                     (0, 0, 0), swapRB=False, crop=False)
        modelo = self.modelo()
        modelo.setInput(entrada_blob)
        saida = modelo.forward()

        pontos = []       
        for i in range(0,22):
            mapa_confianca = saida[0, i, :, :]
            mapa_confianca = cv2.resize(mapa_confianca, (imagem_largura, imagem_altura))
            _, confianca, _, ponto = cv2.minMaxLoc(mapa_confianca)


            if confianca > 0.1:
                cv2.circle(imagem_copia, (int(ponto[0]), int(ponto[1])), 5, (0,0,0), 
                        thickness=4, lineType=cv2.FILLED)
                cv2.putText(imagem_copia, ' ' + (str(int(ponto[0]))) + ',' + 
                            str(int(ponto[1])), (int(ponto[0]), int(ponto[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 0, lineType=cv2.LINE_AA)

                cv2.circle(imagem, (int(ponto[0]), int(ponto[1])), 4,
                        (0,0,0),
                        thickness=4, lineType=cv2.FILLED)
                cv2.putText(imagem, ' ' + str(i), (int(ponto[0]), 
                                                        int(ponto[1])), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (0,0,0),
                            1, lineType=cv2.LINE_AA)

                pontos.append((int(ponto[0]), int(ponto[1])))

            else:
                pontos.append((0, 0))

        for par in [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], 
                    [0, 9], [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15],
                    [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]:


            if pontos[par[0]] != (0, 0) and pontos[par[1]] != (0, 0):
                cv2.line(imagem_copia, pontos[par[0]], pontos[par[1]],  
                        2, lineType=cv2.LINE_AA)
                cv2.line(imagem, pontos[par[0]], pontos[par[1]],(0, 0, 255), 2, 
                        lineType=cv2.LINE_AA)

        posicao.posicoes = []

        # Dedo polegar
        posicao.verificar_posicao_DEDOS(pontos[1:5], 'polegar', altura.verificar_altura_MAO(pontos))

        # Dedo indicador
        posicao.verificar_posicao_DEDOS(pontos[5:9], 'indicador', altura.verificar_altura_MAO(pontos))

        # Dedo médio
        posicao.verificar_posicao_DEDOS(pontos[9:13], 'medio', altura.verificar_altura_MAO(pontos))

        # Dedo anelar
        posicao.verificar_posicao_DEDOS(pontos[13:17], 'anelar', altura.verificar_altura_MAO(pontos))

        # Dedo mínimo
        posicao.verificar_posicao_DEDOS(pontos[17:21], 'minimo', altura.verificar_altura_MAO(pontos))
        
        letras = [chr(i) for i in range(65,91)]
        for i, a in enumerate(alfabeto.letras):
            if proximidade.verificar_proximidade_DEDOS(pontos) == alfabeto.letras[i]:
                print(f'Letra : {letras[i]}')
                cv2.putText(imagem, ' ' + letras[i], (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (69,69,69),
                            2, lineType=cv2.LINE_AA)


        # plt.figure(figsize= [14,10])
        # plt.axis("off")
        # plt.imshow(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))
        # plt.show()




class Montagem():
    def gabinete(self,conteudo):
        return conteudo

if __name__ == '__main__':

    arquivo_proto = "pose_deploy.prototxt"
   
    arquivo_pesos = "pose_iter_102000.caffemodel"
  
    imagem = ""
    imagem2 = ""
  
    fabricar = Montagem()
    
    libras = RedeNeural(imagem, arquivo_proto,arquivo_pesos)
    obj = fabricar.gabinete(libras)
    obj.processamento()



    libras2 = RedeNeural(imagem2, arquivo_proto,arquivo_pesos)
    obj2 = fabricar.gabinete(libras2)
    obj2.processamento()
