#include <iostream>
#include <thread>
#include <mutex>
#include <chrono>
#include <queue>
#include <condition_variable>
#include <cstdlib>
#include <cmath>


using namespace std;

mutex mtx;

// zad 1
int counter = 0;

// zad 2
queue<int> buffer;
condition_variable cv;
bool done = false;
const int MAX_ITEMS = 100;

// zad 3


void wf1(int id) {
    for (int i = 0; i < 10000; ++i) {
        mtx.lock();
        if(id % 2 == 0) {
            counter++;
        } else{
            counter--;
        }
        mtx.unlock();
    }
}

void zad1() {
    cout << "Uruchamianie wątków..." << endl;

    thread thread1(wf1, 1);
    thread thread2(wf1, 2);

    thread1.join();
    thread2.join();

    cout << "Watki zakonczyly prace. ostateczna wartosc licznika to:" << counter << endl;

}
void producer() {
    for (int i = 1; i <= MAX_ITEMS; ++i) {
        unique_lock<mutex> lock(mtx);  // Blokowanie mutexa
        buffer.push(rand()%50);
        cout << "Producent: Wyprodukowano " << i << endl;
        cv.notify_one(); // Powiadomienie konsumenta
    }

    // Ustawienie flagi zakończenia i powiadomienie konsumenta
    unique_lock<mutex> lock(mtx);
    done = true;
    cv.notify_one();
}

// Funkcja konsumenta
void consumer() {
    while (true) {
        unique_lock<mutex> lock(mtx);
        cv.wait(lock, [] { return !buffer.empty() || done; }); // Oczekiwanie na dane

        if (buffer.empty() && done) {
            break; // Jeśli produkcja zakończona i kolejka pusta -> zakończ działanie
        }

        int item = buffer.front();
        buffer.pop();
        cout << "Konsument: Przetworzono " << item << endl << "Wynik: " << pow(item, rand()%3) << endl;
    }
}

void zad2() {
    thread producerThread(producer);
    thread consumerThread(consumer);

    producerThread.join();
    consumerThread.join();

    cout << "Zakończono produkcję i konsumpcję." << endl;
}

void zad3() {

}

int main() {
    //zad1();
    //zad2();

    return 0;
}